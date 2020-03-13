from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np

import config as configs
parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

from util import Snapshot, Summary, InitNodes, StopWatch
#from dali_util import get_rec_iter
import ofrecord_util
from job_function_util import get_train_config, get_val_config
import oneflow as flow
#import vgg_model
import resnet_model
#import alexnet_model


total_device_num = args.num_nodes * args.gpu_num_per_node
train_batch_size = total_device_num * args.batch_size_per_device
val_batch_size = total_device_num * args.val_batch_size_per_device
(C, H, W) = args.image_shape
epoch_size = math.ceil(args.num_examples / train_batch_size)
num_val_steps = args.num_val_examples / val_batch_size

summary = Summary(args.log_dir, args)
timer = StopWatch()

model_dict = {
    "resnet50": resnet_model.resnet50,
    #"vgg16": vgg_model.vgg16,
    #"alexnet": alexnet_model.alexnet,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)

@flow.function(get_train_config(args))
def TrainNet():
    if args.train_data_dir:
        assert os.path.exists(args.train_data_dir)
        print("Loading data from {}".format(args.train_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_training(args)
    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    #loss = flow.math.reduce_mean(loss)
    flow.losses.add_loss(loss)
    softmax = flow.nn.softmax(logits)
    outputs = {"loss": loss, "softmax":softmax, "labels": labels}
    return outputs


@flow.function(get_val_config(args))
def InferenceNet():
    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)
    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    outputs = {"softmax":softmax, "labels": labels}
    return outputs#(softmax, labels)


def acc_acc(step, predictions):
    classfications = np.argmax(predictions['softmax'].ndarray(), axis=1)
    labels = predictions['labels'].reshape(-1)
    if step == 0:
        main.correct = 0.0
        main.total = 0.0
    else:
        main.correct += np.sum(classfications == labels);
        main.total += len(labels)


def train_callback(epoch, step):
    def callback(train_outputs):
        acc_acc(step, train_outputs)
        loss = train_outputs['loss'].mean()
        summary.scalar('loss', loss, step)
        #summary.scalar('learning_rate', train_outputs['lr'], step)
        if (step-1) % args.loss_print_every_n_iter == 0:
            throughput = args.loss_print_every_n_iter * train_batch_size / timer.split()
            accuracy = main.correct/main.total
            print("epoch {}, iter {}, loss: {:.6f}, accuracy: {:.6f}, samples/s: {:.3f}".format(
                   epoch, step-1, loss, accuracy, throughput))
            summary.scalar('train_accuracy', accuracy, step)
            main.correct = 0.0
            main.total = 0.0
    return callback


def do_predictions(epoch, predict_step, predictions):
    acc_acc(predict_step, predictions)
    if predict_step + 1 == num_val_steps:
        assert main.total > 0
        summary.scalar('top1_accuracy', main.correct/main.total, epoch)
        #summary.scalar('top1_correct', main.correct, epoch)
        #summary.scalar('total_val_images', main.total, epoch)
        print("epoch {}, top 1 accuracy: {:.6f}, time: {:.2f}".format(epoch,
              main.correct/main.total, timer.split()))


def predict_callback(epoch, predict_step):
    def callback(predictions):
        do_predictions(epoch, predict_step, predictions)
    return callback


def main():
    InitNodes(args)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    timer.start()
    for epoch in range(args.num_epochs):
        tic = time.time()
        print('Starting epoch {} at {:.2f}'.format(epoch, tic))
        for i in range(epoch_size):
            TrainNet().async_get(train_callback(epoch, i))
        #    if i > 30:#debug
        #        break
        #break
        print('epoch {} training time: {:.2f}'.format(epoch, time.time() - tic))
        if args.val_data_dir:
            tic = time.time()
            for i in range(num_val_steps):
                InferenceNet().async_get(predict_callback(epoch, i))

        summary.save()
        snapshot.save('epoch_{}'.format(epoch+1))


if __name__ == "__main__":
    main()
