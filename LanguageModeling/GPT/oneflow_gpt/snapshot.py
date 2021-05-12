import os
import re
import glob
import operator
import oneflow as flow


class Snapshot(object):
    def __init__(
        self,
        load_dir=None,
        save_dir=None,
        save_interval=0,
        total_iters=0,
        save_last=False,
        save_init=False,
    ):
        self.load_dir_ = load_dir
        self.save_dir_ = save_dir
        self.save_interval_ = save_interval
        self.total_iters_ = total_iters
        self.save_last_ = save_last
        self.save_init_ = save_init
        self.checkpoint_ = flow.train.CheckPoint()

        if load_dir is None:
            self.iter_ = 0
            self.checkpoint_.init()
        else:
            self.iter_, snapshot_dir = self._find_max_iter_snapshot_from_load_dir()
            if snapshot_dir is not None:
                print(f"Loading model from {snapshot_dir}")
                self.checkpoint_.load(snapshot_dir)

        self._check_save_dir_snapshot_existence(self.iter_)

    def _extract_iter_from_snapshot_dirname(self, s):
        itr_str = re.findall(r"\d+", s)
        itr = list(map(int, itr_str))
        assert len(itr) > 0
        return itr[0]

    def _collect_snapshot2iter(self, basedir):
        snapshot_dirs = glob.glob(f"{basedir}/iter*_snapshot")
        snapshot2iter = dict()
        for s_dir in snapshot_dirs:
            assert os.path.isdir(s_dir)
            s = os.path.basename(s_dir)
            snapshot2iter[s_dir] = self._extract_iter_from_snapshot_dirname(s)
        return snapshot2iter

    def _check_save_dir_snapshot_existence(self, start_iter):
        snapshot2iter = self._collect_snapshot2iter(self.save_dir_)
        for s, i in snapshot2iter.items():
            if self.save_init_ and i == 0:
                raise ValueError(f"{s} already exist")

            if self.save_last_ and i == self.total_iters_:
                raise ValueError(f"{s} already exist")

            if (
                i > start_iter
                and self.save_interval_ > 0
                and (i - start_iter) % self.save_interval_ == 0
                and i <= self.total_iters_
            ):
                raise ValueError(f"{s} already exist")

    def _find_max_iter_snapshot_from_load_dir(self):
        snapshot2iter = self._collect_snapshot2iter(self.load_dir_)
        if len(snapshot2iter) == 0:
            return 0, None

        s, i = max(snapshot2iter.items(), key=operator.itemgetter(1))
        return i, s

    @property
    def iter(self):
        return self.iter_

    def save(self, name):
        if self.save_dir_ is None:
            return

        save_path = os.path.join(self.save_dir_, name)
        if os.path.exists(save_path):
            return

        os.makedirs(save_path)
        print(f"Saving model to {save_path}")
        self.checkpoint_.save(save_path)

    def step(self):
        if self.iter_ == 0 and self.save_init_:
            self.save("iter0_snapshot")

        self.iter_ += 1

        if self.save_interval_ > 0 and self.iter_ % self.save_interval_ == 0:
            self.save(f"iter{self.iter_}_snapshot")

        if self.iter_ == self.total_iters_ and self.save_last_:
            self.save(f"iter{self.total_iters_}_snapshot")
