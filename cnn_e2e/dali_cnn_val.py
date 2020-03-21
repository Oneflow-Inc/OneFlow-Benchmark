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
num_val_steps = int(args.num_val_examples / val_batch_size)


model_dict = {
    "resnet50": resnet_model.resnet50,
    #"vgg16": vgg_model.vgg16,
    #"alexnet": alexnet_model.alexnet,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)
@flow.function(get_val_config(args))
def InferenceNet(images=flow.FixedTensorDef((val_batch_size, H, W, C), dtype=flow.float),
                 labels=flow.FixedTensorDef((val_batch_size, ), dtype=flow.int32)):
    logits = model_dict[args.model](images)
    softmax = flow.nn.softmax(logits)
    outputs = {"softmax":softmax, "labels": labels}
    return outputs#(softmax, labels)


def main():
    InitNodes(args)
    assert args.model_load_dir, 'must have model load dir'

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)

    train_data_iter, val_data_iter = get_rec_iter(args, True)
    for epoch in range(args.num_epochs):
        model_load_dir = os.path.join(args.model_load_dir, 'snapshot_epoch_{}'.format(epoch+1))
        snapshot = Snapshot(args.model_save_dir, model_load_dir)
        metric = Metric(desc='validataion', calculate_batches=num_val_steps, summary=summary,
                        save_summary_steps=num_val_steps, batch_size=val_batch_size)
        val_data_iter.reset()
        for i in range(num_val_steps):
            images, labels = batches
            InferenceNet(images, labels).async_get(predict_callback(epoch, i))
        summary.save()


if __name__ == "__main__":
    main()
