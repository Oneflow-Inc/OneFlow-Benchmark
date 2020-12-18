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
import time
import math
import numpy as np

import config as configs
parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

from util import Snapshot, InitNodes, Metric
import ofrecord_util
from job_function_util import get_train_config, get_val_config
import oneflow as flow
import vgg_model
import resnet_model
import resnext_model
import alexnet_model
import mobilenet_v2_model


total_device_num = args.num_nodes * args.gpu_num_per_node
val_batch_size = total_device_num * args.val_batch_size_per_device
(C, H, W) = args.image_shape
num_val_steps = int(args.num_val_examples / val_batch_size)


model_dict = {
    "resnet50": resnet_model.resnet50,
    "vgg": vgg_model.vgg16bn,
    "alexnet": alexnet_model.alexnet,
    "mobilenetv2": mobilenet_v2_model.Mobilenet,
    "resnext50": resnext_model.resnext50,
}


flow.config.gpu_device_num(args.gpu_num_per_node)
#flow.config.enable_debug_mode(True)
@flow.global_function("predict", get_val_config(args))
def InferenceNet():
    assert os.path.exists(args.val_data_dir)
    print("Loading data from {}".format(args.val_data_dir))
    (labels, images) = ofrecord_util.load_imagenet_for_validation(args)

    logits = model_dict[args.model](images, args)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)
    assert args.model_load_dir, 'Must have model load dir!'

    flow.env.log_dir(args.log_dir)
    # snapshot = Snapshot(args.model_save_dir, args.model_load_dir)
    print("Restoring model from {}.".format(args.model_load_dir))
    checkpoint = flow.train.CheckPoint()
    checkpoint.load(args.model_load_dir)
    metric = Metric(desc='validation', calculate_batches=num_val_steps, 
                    batch_size=val_batch_size)
    
    for i in range(args.num_epochs):
        for j in range(num_val_steps):
            InferenceNet().async_get(metric.metric_cb(0, j))


if __name__ == "__main__":
    main()
