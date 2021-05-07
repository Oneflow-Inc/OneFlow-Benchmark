import os
import numpy as np
import oneflow as flow

from .config import get_args


# Warning: this impl rely on specified OneFlow version saving train step policy
def _load_saved_iter(load_dir, train_func_name):
    iter = 0
    train_step_path = f"{load_dir}/System-Train-TrainStep-{train_func_name}/out"
    with open(train_step_path, "rb") as f:
        iter = np.frombuffer(f.read(), dtype=np.int64).item()

    return iter


class Snapshot(object):
    def __init__(self, train_func_name):
        self.checkpoint_ = flow.train.CheckPoint()
        args = get_args()
        if args.load is None:
            self.checkpoint_.init()
            self.iter_ = 0
        else:
            self.checkpoint_.load(args.load)
            self.iter_ = _load_saved_iter(args.load, train_func_name)

        self.save_dir_ = args.save
        self.save_interval_ = args.save_interval or 0
        self.save_last_ = args.save_last
        self.train_iters_ = args.train_iters

    @property
    def iter(self):
        return self.iter_

    def save(self, name):
        assert self.save_dir_ is not None
        save_path = os.path.join(self.save_dir_, name)
        if os.path.exists(save_path):
            raise ValueError("snapshot path '{save_path}' already exist")

        os.makedirs(save_path)
        print(f"Saving model to {save_path}")
        self.checkpoint_.save(save_path)

    def step(self):
        self.iter_ += 1

        if (
            self.save_dir_ is not None
            and self.save_interval_ > 0
            and self.iter_ % self.save_interval_ == 0
        ):
            self.save(f"iter{iter}_snapshot")

        if self.iter_ == self.train_iters_ and self.save_last_:
            self.save("last_snapshot")
