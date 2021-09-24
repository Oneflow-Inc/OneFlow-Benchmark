"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import math
import oneflow.compatible.single_client as flow
import ofrecord_util
import optimizer_util
import config as configs
from util import Snapshot, InitNodes, Metric
from job_function_util import get_train_config, get_val_config
import resnet_model
import resnext_model
import vgg_model
import alexnet_model
import inception_model
import mobilenet_v2_model
from util import build_watch_cb, build_watch_diff_cb

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
    "vgg": vgg_model.vgg16bn,
    "alexnet": alexnet_model.alexnet,
    "inceptionv3": inception_model.inceptionv3,
    "mobilenetv2": mobilenet_v2_model.Mobilenet,
    "resnext50": resnext_model.resnext50,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)

if args.use_fp16 and args.num_nodes * args.gpu_num_per_node > 1:
    flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)

if args.nccl_fusion_threshold_mb:
    flow.config.collective_boxing.nccl_fusion_threshold_mb(
        args.nccl_fusion_threshold_mb
    )

if args.nccl_fusion_max_ops:
    flow.config.collective_boxing.nccl_fusion_max_ops(args.nccl_fusion_max_ops)

if args.num_nodes > 1 and args.use_rdma:
    flow.config.use_rdma(True)

def label_smoothing(labels, classes, eta, dtype):
    assert classes > 0
    assert eta >= 0.0 and eta < 1.0
    return flow.one_hot(
        labels,
        depth=classes,
        dtype=dtype,
        on_value=1 - eta + eta / classes,
        off_value=eta / classes,
    )


@flow.global_function("train", get_train_config(args))
def TrainNet():
    if args.train_data_dir:
        assert os.path.exists(args.train_data_dir)
        print("Loading data from {}".format(args.train_data_dir))
        #(labels, images) = ofrecord_util.load_imagenet_for_training(args)
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)
    logits = model_dict[args.model](images, args)
    flow.watch(logits, build_watch_cb('logits'))
    flow.watch_diff(logits, build_watch_diff_cb('logits_grad'))
    if args.label_smoothing > 0:
        one_hot_labels = label_smoothing(
            labels, args.num_classes, args.label_smoothing, logits.dtype
        )
        loss = flow.nn.softmax_cross_entropy_with_logits(
            one_hot_labels, logits, name="softmax_loss"
        )
    else:
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    loss = flow.math.reduce_mean(loss)
    predictions = flow.nn.softmax(logits)
    outputs = {"loss": loss, "predictions": predictions, "labels": labels, 'images': images, 'logits': logits}

    # set up warmup,learning rate and optimizer
    optimizer_util.set_up_optimizer(loss, args)
    return outputs


@flow.global_function("predict", get_val_config(args))
def InferenceNet():
    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](images, args)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)
    flow.env.log_dir(args.log_dir)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    print(" {} iter per epoch...".format(epoch_size))

    for epoch in range(args.num_epochs):
        metric = Metric(
            desc="train",
            calculate_batches=args.loss_print_every_n_iter,
            batch_size=train_batch_size,
            loss_key="loss",
        )
        for i in range(epoch_size):
            # TrainNet().async_get(metric.metric_cb(epoch, i))
            a = TrainNet().get()
            # snapshot.save("epoch_{}_iter{}".format(epoch, i))
            print('loss:', a['loss'].numpy())
            if i>=100:
                break
        break

        if args.val_data_dir:
            metric = Metric(
                desc="validation",
                calculate_batches=num_val_steps,
                batch_size=val_batch_size,
            )
            for i in range(num_val_steps):
                InferenceNet().async_get(metric.metric_cb(epoch, i))
        snapshot.save("epoch_{}".format(epoch))


if __name__ == "__main__":
    main()
