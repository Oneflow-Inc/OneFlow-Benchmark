from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np
import logging

import oneflow as flow

import data_loader
import vgg_model
import resnet_model
import alexnet_model

import config as configs
from util import Snapshot, Summary, print_args, make_lr, nodes_init
from dali import get_rec_iter


parser = configs.get_parser()
args = parser.parse_args()

total_device_num = args.num_nodes * args.gpu_num_per_node
train_batch_size = total_device_num * args.batch_size_per_device
val_batch_size = total_device_num * args.val_batch_size_per_device
(H, W, C) = (args.image_size, args.image_size, 3)
epoch_size = math.ceil(args.num_examples / train_batch_size)
num_train_batches = epoch_size * args.num_epochs
num_warmup_batches = epoch_size * args.warmup_epochs

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
    "momentum-cosine-decay": {
        "momentum_conf": {"beta": 0.875},
        "warmup_conf": {"linear_conf": {"warmup_batches":num_warmup_batches, "start_multiplier":0}},
        "learning_rate_decay": {"cosine_conf": {"decay_batches": num_train_batches - num_warmup_batches}},
    },
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)
def get_train_config():
    train_config = flow.function_config()
    train_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    train_config.default_data_type(flow.float)
    train_config.train.primary_lr(args.learning_rate)
    train_config.disable_all_reduce_sequence(True)
    #train_config.all_reduce_group_min_mbyte(8)
    #train_config.all_reduce_group_num(128)
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
    return train_config


@flow.function(get_train_config())
def NumpyTrainNet(images=flow.FixedTensorDef((train_batch_size, H, W, C), dtype=flow.float),
                  labels=flow.FixedTensorDef((train_batch_size, 1), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    softmax = flow.nn.softmax(logits)
    outputs = {"loss": loss, "softmax":softmax, "labels": labels}
    return outputs


def get_val_config():
    val_config = flow.function_config()
    val_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    val_config.default_data_type(flow.float)
    return val_config


@flow.function(get_val_config())
def InferenceNet(images=flow.FixedTensorDef((val_batch_size, H, W, C), dtype=flow.float),
                 labels=flow.FixedTensorDef((val_batch_size, 1), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    outputs = {"softmax":softmax, "labels": labels}
    return outputs#(softmax, labels)

def acc_acc(step, predictions):
    classfications = np.argmax(predictions['softmax'].ndarray(), axis=1)
    labels = predictions['labels'].reshape(-1)
    #print('cls')
    #print(classfications)
    #print('labels')
    #print(labels.reshape(-1))
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
            accuracy = main.correct/main.total
            print("epoch {}, iter {}, loss: {:.6f}, accuracy: {:.6f}".format(epoch, step-1, loss,
                                                                             accuracy))
            summary.scalar('train_accuracy', accuracy, step)
            main.correct = 0.0
            main.total = 0.0
            #exit()
    return callback


def do_predictions(epoch, predict_step, predictions):
    acc_acc(predict_step, predictions)
    #classfications = np.argmax(predictions['softmax'].ndarray(), axis=1)
    #labels = predictions['labels']
    #if predict_step == 0:
    #    main.correct = 0.0
    #    main.total = 0.0
    #else:
    #    main.correct += np.sum(classfications == labels);
    #    main.total += len(labels)
    if predict_step + 1 == args.val_step_num:
        assert main.total > 0
        summary.scalar('top1_accuracy', main.correct/main.total, epoch)
        #summary.scalar('top1_correct', main.correct, epoch)
        #summary.scalar('total_val_images', main.total, epoch)
        print("epoch {}, top 1 accuracy: {:.6f}".format(epoch, main.correct/main.total))


def predict_callback(epoch, predict_step):
    def callback(predictions):
        do_predictions(epoch, predict_step, predictions)
    return callback


def main():
    print_args(args)
    nodes_init(args)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    train_data_iter, val_data_iter = get_rec_iter(args, True)
    for epoch in range(args.num_epochs):
        print('Starting epoch {}'.format(epoch))
        tic = time.time()
        train_data_iter.reset()
        for i, batches in enumerate(train_data_iter):
            assert len(batches) == 1
            images, labels = batches[0]
            NumpyTrainNet(images, labels.astype(np.int32)).async_get(train_callback(epoch, i))
        print(time.time() - tic)
        if args.data_val:
            tic = time.time()
            val_data_iter.reset()
            for i, batches in enumerate(val_data_iter):
                assert len(batches) == 1
                images, labels = batches[0]
                InferenceNet(images, labels.astype(np.int32)).async_get(predict_callback(epoch, i))
            print(time.time() - tic)


    #step += 1
    #for predict_step in range(args.val_step_num):
    #    do_predictions(step, predict_step, InferenceNet().get()) #use sync mode
    #snapshot.save(step)
    summary.save()


if __name__ == "__main__":
    main()
