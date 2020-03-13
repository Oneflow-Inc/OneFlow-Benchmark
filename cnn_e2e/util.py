from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pandas as pd
from datetime import datetime
import oneflow as flow


def InitNodes(args):
    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)


class Snapshot:
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        self._check_point = flow.train.CheckPoint()
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            self._check_point.load(model_load_dir)
        else:
            print("Init model on demand.")
            self._check_point.init()

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)


class Summary():
    def __init__(self, log_dir, config):
        self._log_dir = log_dir
        self._metrics = pd.DataFrame({"iter": 0, "legend": "cfg", "note": str(config)}, index=[0])

    def scalar(self, legend, value, step=-1):
        # TODO: support rank(which device/gpu)
        df = pd.DataFrame(
            {"iter": step, "legend": legend, "value": value, "rank": 0, "time": time.time()},
            index=[0])
        self._metrics = pd.concat([self._metrics, df], axis=0, sort=False)

    def save(self):
        save_path = os.path.join(self._log_dir, "summary.csv")
        self._metrics.to_csv(save_path, index=False)
        print("saved: {}".format(save_path))


class StopWatch:
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

