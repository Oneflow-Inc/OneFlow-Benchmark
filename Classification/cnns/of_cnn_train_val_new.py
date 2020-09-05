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
import oneflow as flow
import ofrecord_util
import config as configs
from util import Snapshot, Summary, InitNodes, Metric

import resnet_model
import resnext_model
import vgg_model
import alexnet_model
import inception_model
import mobilenet_v2_model

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
#flow.config.enable_debug_mode(True)


def _default_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    return config


def get_train_config(args):
    train_config = _default_config(args)
    train_config.cudnn_conv_heuristic_search_algo(False)


    train_config.prune_parallel_cast_ops(True)
    train_config.enable_inplace(True)
    return train_config


def get_val_config(args):
    return _default_config(args)


def label_smoothing(labels, classes, eta, dtype):
    assert classes > 0
    assert eta >= 0.0 and eta < 1.0

    return flow.one_hot(labels, depth=classes, dtype=dtype,
                        on_value=1 - eta + eta / classes, off_value=eta/classes)


@flow.global_function("train", get_train_config(args))
def TrainNet():
    if args.train_data_dir:
        assert os.path.exists(args.train_data_dir)
        print("Loading data from {}".format(args.train_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_training(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)
    logits = model_dict[args.model](images,
                                    need_transpose=False if args.train_data_dir else True,
                                    )
    if args.label_smoothing > 0:
        one_hot_labels = label_smoothing(labels, args.num_classes, args.label_smoothing, logits.dtype)
        loss = flow.nn.softmax_cross_entropy_with_logits(one_hot_labels, logits, name="softmax_loss")
    else:
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    loss = flow.math.reduce_mean(loss)    
    flow.losses.add_loss(loss)
    predictions = flow.nn.softmax(logits)
    outputs = {"loss": loss, "predictions": predictions, "labels": labels}

    # set up warmup,learning rate and optimizer
    set_up_optimizer(loss, args)
    return outputs


def set_up_optimizer(loss, args):
    # set up warmup,learning rate and optimizer
    print("set_up_optimizer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n args.gradient_clipping, args.warmup_epochs, args.learning_rate", 
        args.gradient_clipping,  args.warmup_epochs, args.learning_rate)
        
    batches_per_epoch = math.ceil(args.num_examples / train_batch_size)
    warmup_batches = batches_per_epoch * args.warmup_epochs
    lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
        base_lr=args.learning_rate, 
        boundaries=[30, 60, 80], 
        scale=[0.1, 0.01, 0.001], 
        warmup=flow.optimizer.warmup.linear(warmup_batches, 0)
        )
    flow.optimizer.SGD(lr_scheduler,
      momentum=args.mom,
      grad_clipping= flow.optimizer.grad_clipping.by_global_norm(args.gradient_clipping) 
          if args.gradient_clipping>0.0  else None
    ).minimize(loss)


@flow.global_function("predict", get_val_config(args))
def InferenceNet():
    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)

    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](
        images, need_transpose=False if args.val_data_dir else True)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)
    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

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


if __name__ == "__main__":
    main()
