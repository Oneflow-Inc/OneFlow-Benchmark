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
import numpy as np
import pandas as pd
from datetime import datetime
import oneflow as flow


def InitNodes(args):
    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        flow.env.ctrl_port(args.ctrl_port)
        nodes = []
        for ip in args.node_ips[:args.num_nodes]:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)


class Snapshot(object):
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        self._check_point = flow.train.CheckPoint()
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            self._check_point.load(model_load_dir)
        else:
            self._check_point.init()
            self.save('initial_model')
            print("Init model on demand.")

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


def match_top_k(predictions, labels, top_k=1):
    max_k_preds = np.argpartition(predictions.numpy(), -top_k)[:, -top_k:]
    match_array = np.logical_or.reduce(max_k_preds == labels.reshape((-1, 1)), axis=1)
    num_matched = match_array.sum()
    return num_matched, match_array.shape[0]


class Metric(object):
    def __init__(self, desc='train', calculate_batches=-1, batch_size=256, top_k=5, 
                 prediction_key='predictions', label_key='labels', loss_key=None):
        self.desc = desc
        self.calculate_batches = calculate_batches
        self.top_k = top_k
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.loss_key = loss_key
        if loss_key:
            self.fmt = "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"
        else:
            self.fmt = "{}: epoch {}, iter {}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        self.top_1_num_matched = 0
        self.top_k_num_matched = 0
        self.num_samples = 0.0

    def metric_cb(self, epoch, step):
        def callback(outputs):
            if step == 0: self._clear()
            if self.prediction_key:
                num_matched, num_samples = match_top_k(outputs[self.prediction_key],
                                                       outputs[self.label_key])
                self.top_1_num_matched += num_matched
                num_matched, _ = match_top_k(outputs[self.prediction_key],
                                             outputs[self.label_key], self.top_k)
                self.top_k_num_matched += num_matched
            else:
                num_samples = outputs[self.label_key].shape[0]

            self.num_samples += num_samples

            if (step + 1) % self.calculate_batches == 0:
                throughput = self.num_samples / self.timer.split()
                if self.prediction_key:
                    top_1_accuracy = self.top_1_num_matched / self.num_samples
                    top_k_accuracy = self.top_k_num_matched / self.num_samples
                else:
                    top_1_accuracy = 0.0
                    top_k_accuracy = 0.0

                if self.loss_key:
                    loss = outputs[self.loss_key].mean()
                    print(self.fmt.format(self.desc, epoch, step + 1, loss, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())
                else:
                    print(self.fmt.format(self.desc, epoch, step + 1, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())

                self._clear()

        return callback


