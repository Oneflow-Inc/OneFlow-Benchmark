from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np

import config as configs
parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

from util import Snapshot, Summary, InitNodes, Metric
from dali_util import get_rec_iter
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


model_dict = {
    "resnet50": resnet_model.resnet50,
    #"vgg16": vgg_model.vgg16,
    #"alexnet": alexnet_model.alexnet,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)

@flow.function(get_train_config(args))
def TrainNet(images=flow.FixedTensorDef((train_batch_size, H, W, C), dtype=flow.float),
             labels=flow.FixedTensorDef((train_batch_size, ), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    loss = flow.math.reduce_mean(loss)
    flow.losses.add_loss(loss)
    predictions = flow.nn.softmax(logits)
    outputs = {"loss": loss, "predictions":predictions, "labels": labels}
    return outputs


@flow.function(get_val_config(args))
def InferenceNet(images=flow.FixedTensorDef((val_batch_size, H, W, C), dtype=flow.float),
                 labels=flow.FixedTensorDef((val_batch_size, ), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions":predictions, "labels": labels}
    return outputs#(softmax, labels)


def main():
    InitNodes(args)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    train_data_iter, val_data_iter = get_rec_iter(args, True)
    for epoch in range(args.num_epochs):
        metric = Metric(desc='train', calculate_batches=args.loss_print_every_n_iter,
                        summary=summary, save_summary_steps=epoch_size,
                        batch_size=train_batch_size, loss_key='loss')
        train_data_iter.reset()
        for i, batches in enumerate(train_data_iter):
            images, labels = batches
            TrainNet(images, labels).async_get(metric.metric_cb(epoch, i))
        #    if i > 30:#debug
        #        break
        #break
        if args.data_val:
            metric = Metric(desc='validation', calculate_batches=num_val_steps, summary=summary,
                            save_summary_steps=num_val_steps, batch_size=val_batch_size)
            val_data_iter.reset()
            for i, batches in enumerate(val_data_iter):
                images, labels = batches
                InferenceNet(images, labels).async_get(metric.metric_cb(epoch, i))

        snapshot.save('epoch_{}'.format(epoch))


if __name__ == "__main__":
    main()
