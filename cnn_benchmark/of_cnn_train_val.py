from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

import oneflow as flow

import data_loader
import vgg_model
import resnet_model
import alexnet_model

import config as configs
from util import Snapshot, Summary, print_args, make_lr


parser = configs.get_parser()
#args = parser.parse_known_args()[0]
args = parser.parse_args()

summary = Summary(args.log_dir, args)


model_dict = {
    "resnet50": resnet_model.resnet50,
    "vgg16": vgg_model.vgg16,
    "alexnet": alexnet_model.alexnet,
}

optimizer_dict = {
    "sgd": {"naive_conf": {}},
    "adam": {"adam_conf": {"beta1": 0.9}},
    "momentum": {"momentum_conf": {"beta": 0.9}},
    "momentum-decay": {
        "momentum_conf": {"beta": 0.9},
        "learning_rate_decay": {
            "polynomial_conf": {"decay_batches": 300000, "end_learning_rate": 0.0001,},
        },
    },
}

#        "warmup_conf": {"linear_conf": {"warmup_batches":10000, "start_multiplier":0}},


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)
def get_train_config():
    train_config = flow.function_config()
    train_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    train_config.default_data_type(flow.float)
    train_config.train.primary_lr(args.learning_rate)
    train_config.disable_all_reduce_sequence(True)
    train_config.all_reduce_group_min_mbyte(8)
    train_config.all_reduce_group_num(128)
    # train_config.all_reduce_lazy_ratio(0)

    # train_config.enable_nccl_hierarchical_all_reduce(True)
    # train_config.cudnn_buf_limit_mbyte(2048)
    # train_config.concurrency_width(2)
    train_config.all_reduce_group_num(128)
    train_config.all_reduce_group_min_mbyte(8)

    train_config.train.model_update_conf(optimizer_dict[args.optimizer])

    if args.weight_l2:
        train_config.train.weight_l2(args.weight_l2)

    train_config.enable_inplace(True)
    # train_config.ctrl_port(12140)
    return train_config


@flow.function(get_train_config())
def TrainNet():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    if args.data_dir:
        assert os.path.exists(args.data_dir)
        print("Loading data from {}".format(args.data_dir))
        (labels, images) = data_loader.load_imagenet(
            args.data_dir, args.image_size, batch_size, args.data_part_num)
    else:
        print("Loading synthetic data.")
        (labels, images) = data_loader.load_synthetic(args.image_size, batch_size)

    logits = model_dict[args.model](images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    outputs = {"loss": loss}

    #train_config = flow.function_config()
    #print(dir(train_config.train))
    #print(type(train_config.train))
    #step_lr = make_lr("System-Train-TrainStep-TrainNet",
    #    train_config.train.model_update_conf(),
    #    train_config.train.primary_lr(),
    #    train_config.train.secondary_lr())
    #outputs.update(step_lr)

    #lbi = logical_blob_id_util.LogicalBlobId()
    #lbi.op_name = "System-Train-PrimaryLearningRate-Scheduler"
    #lbi.blob_name = "out"
    #lr = remote_blob_util.RemoteBlob(lbi)
    #outputs.update({"lr": lr})
    return outputs


def get_val_config():
    val_config = flow.function_config()
    val_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    val_config.default_data_type(flow.float)
    return val_config


@flow.function(get_val_config())
def InferenceNet():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.val_batch_size_per_device

    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        # TODO: use different image preprocess
        (labels, images) = data_loader.load_imagenet(
            args.val_data_dir, args.image_size, batch_size, args.val_data_part_num)
    else:
        print("Loading synthetic data.")
        (labels, images) = data_loader.load_synthetic(
            args.image_size, batch_size)

    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    return (softmax, labels)


def main():
    print_args(args)
    def train_callback(step):
        def callback(train_outputs):
            loss = train_outputs['loss'].mean()
            summary.scalar('loss', loss, step)
            #summary.scalar('learning_rate', train_outputs['lr'], step)
            if (step-1) % args.loss_print_every_n_iter == 0:
                print("iter {}, loss: {:.6f}".format(step-1, loss))
        return callback

    def do_predictions(step, predict_step, predictions):
        classfications = np.argmax(predictions[0].ndarray(), axis=1)
        labels = predictions[1]
        if predict_step == 0:
            main.correct = 0.0
            main.total = 0.0
        else:
            main.correct += np.sum(classfications == labels);
            main.total += len(labels)
        if predict_step + 1 == args.val_step_num:
            assert main.total > 0
            summary.scalar('top1_accuracy', main.correct/main.total, step)
            #summary.scalar('top1_correct', main.correct, step)
            #summary.scalar('total_val_images', main.total, step)
            print("iter {}, top 1 accuracy: {:.6f}".format(step, main.correct/main.total))

    def predict_callback(step, predict_step):
        def callback(predictions):
            do_predictions(step, predict_step, predictions)
        return callback

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    if args.node_num > 1:
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    total_batch_size = (args.node_num * args.gpu_num_per_node * args.batch_size_per_device)

    for step in range(args.train_step_num):
        # save model every n iter
        if step % args.model_save_every_n_iter == 0:
            # do validation when save model
            if step >= 0: # >=0 will trigger validation at step 0
                for predict_step in range(args.val_step_num):
                    InferenceNet().async_get(predict_callback(step, predict_step))
            snapshot.save(step)

        TrainNet().async_get(train_callback(step+1))

    step += 1
    for predict_step in range(args.val_step_num):
        do_predictions(step, predict_step, InferenceNet().get()) #use sync mode
    snapshot.save(step)
    summary.save()


if __name__ == "__main__":
    main()
