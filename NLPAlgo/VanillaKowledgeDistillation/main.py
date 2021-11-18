import argparse
import numpy as np
import os
import oneflow as flow
import oneflow.typing as tp
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from model import TeacherModel, StudentModel
from util import Snapshot, InitNodes, Metric, CreateOptimizer, GetFunctionConfig
from datetime import datetime
from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="flags for knowledge distillation")
parser.add_argument('--display_step', type=int, default=200)
parser.add_argument('--loss_print_every_n_iter', type=int, default=100)
# parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
parser.add_argument("--model_save_dir", type=str,
        default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
        required=False, help="model save directory")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])
parser.add_argument('--use_fp16', type=str2bool, nargs='?', default='False', const=True,
                    help='use use fp16 or not')
parser.add_argument('--use_xla', type=str2bool, nargs='?', const=True,
                    help='Whether to use use xla')

# Training Parameters
parser.add_argument("--load_teacher_from_checkpoint", action='store_true', help="Whether to training with teacher model")
parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
parser.add_argument('--num_steps', type=int, default=500)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=10)

# Model Parameters
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--dropoutprob', type=float, default=0.25)
parser.add_argument("--weight_decay_rate", type=float, default=0.01, help="weight decay rate")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
args = parser.parse_args()




@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def train_teacher_job(
    images: tp.Numpy.Placeholder((args.batch_size, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((args.batch_size, ), dtype=flow.int32),
):
    model = TeacherModel(
        args=args,
        model_type='teacher',
        X=images,
        Y=labels
    )
    loss, _, __ = model.get_res()
    flow.losses.add_loss(loss)
    # opt = CreateOptimizer(args)
    # opt.minimize(loss)
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return {'loss': loss}


@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def train_student_job(
    images: tp.Numpy.Placeholder((args.batch_size, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((args.batch_size, ), dtype=flow.int32),
    soft_labels: tp.Numpy.Placeholder((args.batch_size, args.num_classes), dtype=flow.float),

):
    model = StudentModel(
        args=args,
        model_type='student',
        X=images,
        Y=labels,
        soft_Y=soft_labels,
        flag=True
    )
    loss, _, __ = model.get_res()
    flow.losses.add_loss(loss)
    # opt = CreateOptimizer(args)
    # opt.minimize(loss)
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return {'loss': loss}


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def eval_teacher_job(
    images: tp.Numpy.Placeholder((args.batch_size, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((args.batch_size, ), dtype=flow.int32),
):
    model = TeacherModel(
        args=args,
        model_type='teacher',
        X=images,
        Y=labels
    )
    _, logits, soft_labels = model.get_res()
    return logits, labels, soft_labels


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def eval_student_job(
    images: tp.Numpy.Placeholder((args.batch_size, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((args.batch_size, ), dtype=flow.int32),
):
    model = StudentModel(
        args=args,
        model_type='student',
        X=images,
        Y=labels,
        flag=False
    )
    _, logits, _ = model.get_res()
    return logits, labels, _


def run_eval_job(dev_job_func, X, Y, model_type='teacher'):
    labels = []
    predictions = []
    for ei, (input_images, input_labels) in enumerate(zip(X, Y)):
        logits, label, _ = dev_job_func(input_images, input_labels).get()
        predictions.extend(list(logits.numpy().argmax(axis=1)))
        labels.extend(list(label))

    def metric_fn(predictions, labels):
        return {
            "accuarcy": accuracy_score(labels, predictions),
            "matthews_corrcoef": matthews_corrcoef(labels, predictions),
            # "precision": precision_score(labels, predictions),
            # "recall": recall_score(labels, predictions),
            # "f1": f1_score(labels, predictions),
        }

    metric_dict = metric_fn(predictions, labels)
    print(model_type, ', '.join('{}: {:.3f}'.format(k, v) for k, v in metric_dict.items()))
    return metric_dict['accuarcy']

def main():
    # 加载数据集
    print('===== loading dataset ... =====')
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(args.batch_size, args.batch_size)

    snapshot = Snapshot(args.model_save_dir)
    if args.model_type == 'teacher':
        print('===== start training teacher model ... =====')
        # 训练teacher模型
        global_step = 0
        best_dev_acc = 0.0
        for epoch in tqdm(range(args.epoch)):
            # print('train_images.shape=', train_images.shape)
            # print('train_labels.shape=', train_labels.shape)
            r = np.random.permutation(len(train_labels))
            train_images_ = train_images[r, :]
            train_labels_ = train_labels[r]
            metric = Metric(desc='teacher-training', print_steps=args.loss_print_every_n_iter,
                            batch_size=args.batch_size, keys=['loss'])

            for (images, labels) in tqdm(zip(train_images_, train_labels_)):
                global_step += 1
                train_teacher_job(images, labels).async_get(metric.metric_cb(global_step, epoch=epoch))

                if (global_step % args.display_step) == 0:
                    # 因为只提供了训练集和测试集的迭代器，因此直接在测试集上进行验证
                    dev_acc = run_eval_job(
                        dev_job_func=eval_teacher_job,
                        X=test_images,
                        Y=test_labels,
                        model_type='teacher')

                    if best_dev_acc < dev_acc:
                        best_dev_acc = dev_acc
                        # 保存最好模型参数
                        print('===== saving teacher model ... =====')
                        snapshot.save("best_teacher_model_dev_{}".format(best_dev_acc), best_dev_acc)
                        # 保存最好的验证准确率

    else:
        print('===== start training student model ... =====')
        # 训练student模型
        # 加载teacher模型
        if args.load_teacher_from_checkpoint:
            snapshot.load(args.load_teacher_checkpoint_dir, args.model_type)
        global_step = 0
        best_dev_acc = 0.0
        for epoch in tqdm(range(args.epoch)):
            r = np.random.permutation(len(train_labels))
            train_images_ = train_images[r, :]
            train_labels_ = train_labels[r]
            metric = Metric(desc='student-training', print_steps=args.loss_print_every_n_iter,
                            batch_size=args.batch_size, keys=['loss'])

            for (images, labels) in tqdm(zip(train_images_, train_labels_)):
                global_step += 1
                # 获得相应样本的soft_label
                soft_labels = labels
                if args.load_teacher_from_checkpoint:
                    logits, _, soft_labels = eval_teacher_job(images, labels).get()
                # print('soft_labels.shape=', soft_labels.shape)
                train_student_job(images, labels, soft_labels.numpy()).async_get(metric.metric_cb(global_step, epoch=epoch))

                if (global_step % args.display_step) == 0:
                    # 因为只提供了训练集和测试集的迭代器，因此直接在测试集上进行验证
                    dev_acc = run_eval_job(
                        dev_job_func=eval_student_job,
                        X=test_images,
                        Y=test_labels,
                        model_type='student')

                    if best_dev_acc < dev_acc:
                        best_dev_acc = dev_acc
                        # 保存最好模型参数
                        print('===== saving student model ... =====')
                        snapshot.save("best_student_model_dev_{}".format(best_dev_acc), best_dev_acc)
                        # 保存最好的验证准确率



if __name__ == '__main__':
    main()