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
from collections import OrderedDict
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


class Summary(object):
    def __init__(self, log_dir, config, filename='summary.csv'):
        self._filename = filename
        self._log_dir = log_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self._metrics = pd.DataFrame({"legend": "cfg", "value": str(config)}, index=[0])

    def scalar(self, legend, iter, value, **kwargs):
        kwargs['legend'] = legend
        kwargs['iter'] = int(iter)
        kwargs['value'] = value
        df = pd.DataFrame(kwargs, index=[0])
        self._metrics = pd.concat([self._metrics, df], axis=0, sort=False)
        self.save()

    def save(self):
        save_path = os.path.join(self._log_dir, self._filename)
        self._metrics.to_csv(save_path, index=False)


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


class Metric(object):
    def __init__(self, summary=None, desc='train', print_steps=-1, batch_size=256, keys=[]):
        r"""accumulate and calculate metric

        Args:
            summary: A `Summary` object to write in.
            desc: `str` general description of the metric to show
            print_steps: `Int` print metrics every nth steps
            batch_size: `Int` batch size per step
            keys: keys in callback outputs
        Returns:
        """
        self.summary = summary
        self.save_summary = isinstance(self.summary, Summary)
        self.desc = desc
        self.print_steps = print_steps
        assert batch_size > 0
        self.batch_size = batch_size

        assert isinstance(keys, (list, tuple))
        self.keys = keys
        self.metric_dict = OrderedDict()
        self.metric_dict['step'] = 0

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        for key in self.keys:
            self.metric_dict[key] = 0.0
        self.metric_dict['throughput'] = 0.0
        self.num_samples = 0.0

    def update_and_save(self, key, value, step, **kwargs):
        self.metric_dict[key] = value
        if self.save_summary:
            self.summary.scalar(self.desc + "_" + key, step, value, **kwargs)

    def metric_cb(self, step=0, **kwargs):
        def callback(outputs):
            if step == 0: self._clear()

            for key in self.keys:
                self.metric_dict[key] += outputs[key].sum()

            self.num_samples += self.batch_size

            if (step + 1) % self.print_steps == 0:
                self.metric_dict['step'] = step
                for k, v in kwargs.items():
                    self.metric_dict[k] = v
                throughput = self.num_samples / self.timer.split()
                self.update_and_save('throughput', throughput, step)
                for key in self.keys:
                    value = self.metric_dict[key] / self.num_samples
                    self.update_and_save(key, value, step, **kwargs)
                print(', '.join(('{}: {}' if type(v) is int else '{}: {:.3f}').format(k, v) \
                                for k, v in self.metric_dict.items()), time.time())
                self._clear()

        return callback

def CreateOptimizer(args):
    warmup_batches = int(args.iter_num * args.warmup_proportion)
    lr_warmup = flow.optimizer.warmup.linear(warmup_batches, 0)
    lr_scheduler = flow.optimizer.PolynomialSchduler(args.learning_rate, args.iter_num, 0.0,
                                                     warmup=lr_warmup)
    return flow.optimizer.AdamW(lr_scheduler, epsilon=1e-6, weight_decay=args.weight_decay_rate,
                                weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
                                grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1.0))

def GetFunctionConfig(args):
    config = flow.function_config()
    config.enable_auto_mixed_precision(args.use_fp16)
    if args.use_xla:
        config.use_xla_jit(True)
    config.enable_fuse_add_to_output(True)
    config.enable_fuse_model_update_ops(True)
    return config

