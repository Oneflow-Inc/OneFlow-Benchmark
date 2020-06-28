from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import oneflow as flow

import ofrecord_util
import config as configs
from util import Snapshot, Summary, InitNodes, Metric
from job_function_util import get_val_config

import resnet_model

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
}

flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)


@flow.global_function(get_val_config(args))
def InferenceNet():
    if args.val_data_dir:
        assert os.path.exists(args.val_data_dir)
        print("Loading data from {}".format(args.val_data_dir))
        (labels, images) = ofrecord_util.load_imagenet_for_validation(args)
    else:
        print("Loading synthetic data.")
        (labels, images) = ofrecord_util.load_synthetic(args)

    logits = model_dict[args.model](images)
    predictions = flow.nn.softmax(logits)
    outputs = {"predictions": predictions, "labels": labels}
    return outputs


def main():
    InitNodes(args)
    assert args.model_load_dir, 'must have model load dir'

    flow.env.grpc_use_no_signal()
    flow.env.log_dir(args.log_dir)

    summary = Summary(args.log_dir, args)

    for epoch in range(args.num_epochs):
        model_load_dir = os.path.join(
            args.model_load_dir, 'snapshot_epoch_{}'.format(epoch))
        snapshot = Snapshot(args.model_save_dir, model_load_dir)
        metric = Metric(desc='validation', calculate_batches=num_val_steps, summary=summary,
                        save_summary_steps=num_val_steps, batch_size=val_batch_size)
        for i in range(num_val_steps):
            InferenceNet().async_get(metric.metric_cb(epoch, i))


if __name__ == "__main__":
    main()
