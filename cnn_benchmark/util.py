import os
import time
import pandas as pd
from datetime import datetime
import oneflow as flow


def nodes_init(args):
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

    def save(self, step):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_%d" % step)
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


def make_lr(train_step_name, model_update_conf, primary_lr, secondary_lr=None):
    # usually, train_step_name is "System-Train-TrainStep-" + train job name
    assert model_update_conf.HasField("learning_rate_decay") or model_update_conf.HasField("warmup_conf"), "only support model update conf with warmup or lr decay for now"
    flow.config.train.train_step_lbn(train_step_name + "-Identity" + "/out")
    secondary_lr_lbn = "System-Train-SecondaryLearningRate-Scheduler/out"
    if secondary_lr is None:
        secondary_lr_lbn = "System-Train-PrimaryLearningRate-Scheduler/out"
    flow.config.train.lr_lbn("System-Train-PrimaryLearningRate-Scheduler/out",
                             "System-Train-SecondaryLearningRate-Scheduler/out")
    # these two lines above must be called before creating any op
    with flow.device_prior_placement("cpu", "0:0"):
        train_step = flow.get_variable(
            name=train_step_name,
            shape=(1,),
            dtype=flow.int64,
            initializer=flow.constant_initializer(0),
            trainable=False
        )
        train_step_id = flow.identity(train_step, name=train_step_name + "-Identity")
        flow.assign(train_step, train_step_id + 1, name=train_step_name + "-Assign")

        primary_lr_blob = flow.schedule(train_step_id, model_update_conf, primary_lr,
            name="System-Train-PrimaryLearningRate-Scheduler")
        secondary_lr_blob = None
        if secondary_lr is None:
            secondary_lr_blob = primary_lr_blob
        else:
            secondary_lr_blob = flow.schedule(train_step_id, model_update_conf, secondary_lr,
                name="System-Train-SecondaryLearningRate-Scheduler")
        assert secondary_lr_blob is not None

        return {
            "train_step": train_step_id,
            "lr": primary_lr_blob,
            "lr2": secondary_lr_blob
        }

def print_args(args):
    print("=".ljust(66, "="))
    print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.model, args.gpu_num_per_node, args.num_nodes))
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

