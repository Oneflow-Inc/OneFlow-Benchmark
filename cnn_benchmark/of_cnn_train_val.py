from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import oneflow as flow

import ofrecord_util
import config as configs
from util import Snapshot, Summary, InitNodes, Metric
from job_function_util import get_train_config, get_val_config
import inception_model
import resnet_model
import vgg_model
import alexnet_model
import time

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)


total_device_num = args.num_nodes * args.gpu_num_per_node
train_batch_size = total_device_num * args.batch_size_per_device
val_batch_size = total_device_num * args.val_batch_size_per_device
(C, H, W) = args.image_shape
epoch_size = math.ceil(args.num_examples / train_batch_size)
num_val_steps = int(args.num_val_examples / val_batch_size)


model_dict = {
    "resnet50": resnet_model.resnet50,
    "vgg16": vgg_model.vgg16,
    "alexnet": alexnet_model.alexnet,
    "inceptionv3": inception_model.inceptionv3,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config
flow.config.enable_debug_mode(True)

if args.use_boxing_v2:
    flow.config.collective_boxing.nccl_fusion_threshold_mb(8)
    flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)


@flow.global_function(get_train_config(args))
def TrainNet():
    if args.train_data_dir:
        assert os.path.exists(args.train_data_dir)
        print("Loading data from {}".format(args.train_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_training(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](
        images, need_transpose=False if args.train_data_dir else True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss")
    loss = flow.math.reduce_mean(loss)
    flow.losses.add_loss(loss)
    predictions = flow.nn.softmax(logits)
    outputs = {"loss": loss, "predictions": predictions, "labels": labels}
    return outputs


@flow.global_function(get_val_config(args))
def InferenceNet():
    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](
        images, need_transpose=False if args.train_data_dir else True)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)
    start_time = time.time()
    for epoch in range(args.num_epochs):
        metric = Metric(desc='train', calculate_batches=args.loss_print_every_n_iter,
                        summary=summary, save_summary_steps=epoch_size,
                        batch_size=train_batch_size, loss_key='loss')
        for i in range(epoch_size):
            TrainNet().async_get(metric.metric_cb(epoch, i))

        if args.val_data_dir:
            metric = Metric(desc='validation', calculate_batches=num_val_steps, summary=summary,
                            save_summary_steps=num_val_steps, batch_size=val_batch_size)
            for i in range(num_val_steps):
                InferenceNet().async_get(metric.metric_cb(epoch, i))
        snapshot.save('epoch_{}'.format(epoch))
        current_time = time.time()
        print("epoch", epoch)
        print("---------------------------------")
        print("Used Time", current_time-start_time)


if __name__ == "__main__":
    main()
