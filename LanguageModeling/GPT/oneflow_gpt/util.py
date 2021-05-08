import time
import numpy as np
import os

from collections import OrderedDict


class _Timer(object):
    def __init__(self):
        pass

    def start(self):
        now = time.perf_counter()
        self.start_ = now
        self.step_ = now

    def step(self):
        now = time.perf_counter()
        duration = now - self.step_
        self.step_ = now
        return duration

    def stop(self):
        self.stop_ = time.perf_counter()

    def cur_step(self):
        return self.step_

    def duration(self):
        return self.stop_ - self.start_


class Metric(object):
    def __init__(
        self,
        print_steps,
        num_samples_per_batch,
        max_samples,
        keys=None,
        print_format="normal",
        nvidia_smi_report_step=10,
        nvidia_smi_report_file=None,
    ):
        r"""accumulate and calculate metric

        Args:
            print_steps: `Int` print metrics every nth steps
            batch_size: `Int` batch size per step
            keys: keys in callback outputs
        Returns:
        """
        self.print_steps_ = print_steps
        self.num_samples_per_micro_batch_ = num_samples_per_batch
        self.max_samples_ = max_samples

        self.nvidia_smi_report_step_ = nvidia_smi_report_step
        self.nvidia_smi_report_file_ = nvidia_smi_report_file

        if keys is None:
            self.keys_ = []
        else:
            self.keys_ = list(keys)

        self.stat_ = OrderedDict()
        self.stat_["batches"] = 0
        self.stat_["acc_batches"] = 0
        self.stat_["micro_batches"] = 0
        self.stat_["acc_micro_batches"] = 0
        self.stat_["samples"] = 0
        self.stat_["timestamp"] = 0.0
        self.stat_["acc_elapsed_time"] = 0.0
        self.stat_["throughput"] = 0.0
        self.stat_["latency"] = 0.0
        for key in self.keys_:
            self.stat_[key] = 0.0

        self.timer_ = _Timer()
        self.timer_.start()

        if print_format == "normal":
            self.print_fn_ = self.step_print
        elif print_format == "table":
            self.print_fn_ = self.step_print_by_table
            self.print_title_ = False
        else:
            raise ValueError("print_format must be <normal|table>")

    def step_print(self):
        record = (
            f"batches={self.stat_['batches']},"
            f"micro_batches={self.stat_['micro_batches']},"
            f"samples={self.stat_['samples']},"
            f"throughput={self.stat_['throughput']:.5f},"
            f"latency={self.stat_['latency']:.5f},"
        )
        for key in self.keys_:
            record += f"{key}={self.stat_[key]:.5f},"

        print(record)

    def step_print_by_table(self):
        title = (
            f"| {'batches'.ljust(8)} "
            f"| {'micro_batches'.ljust(15)} "
            f"| {'samples'.ljust(15)} "
            f"| {'throughput'.ljust(10)} "
            f"| {'latency'.ljust(10)} "
        )
        sep = f"| {'-' * 8} | {'-' * 15} | {'-' * 15} | {'-' * 10} | {'-' * 10} "

        record = (
            f"| {self.stat_['batches']:<8d} "
            f"| {self.stat_['micro_batches']:<15d} "
            f"| {self.stat_['samples']:<15d} "
            f"| {self.stat_['throughput']:<10.5f} "
            f"| {self.stat_['latency']:<10.5f} "
        )

        for key in self.keys_:
            title += f"| {key.ljust(10)} "
            sep += f"| {'-' * 10} "
            record += f"| {self.stat_[key]:<10.5f} "

        title += "|"
        sep += "|"
        record += "|"

        if not self.print_title_:
            print(title)
            print(sep)
            self.print_title_ = True

        print(record)

    def step_stat(self):
        self.stat_["acc_batches"] = 0
        self.stat_["acc_micro_batches"] = 0
        self.stat_["acc_elapsed_time"] = 0

    def metric_cb(self):
        def callback(outputs):
            elapsed_time = self.timer_.step()
            self.stat_["timestamp"] = self.timer_.cur_step()
            self.stat_["acc_elapsed_time"] += elapsed_time

            micro_batches = None
            for key in self.keys_:
                output = outputs[key].numpy()
                assert isinstance(output, np.ndarray)
                if micro_batches is None:
                    micro_batches = output.shape[0]
                else:
                    assert micro_batches == output.shape[0]
                self.stat_[key] += output.sum()

            self.stat_["acc_batches"] += 1
            self.stat_["acc_micro_batches"] += micro_batches

            self.stat_["batches"] += 1
            self.stat_["micro_batches"] += micro_batches
            self.stat_["samples"] += micro_batches * self.num_samples_per_micro_batch_

            if self.stat_["batches"] == self.nvidia_smi_report_step_:
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
                if self.nvidia_smi_report_file_ is not None:
                    cmd += f" -f {self.nvidia_smi_report_file_}"
                os.system(cmd)
                self.print_title_ = False

            if (
                self.stat_["batches"] % self.print_steps_ == 0
                or self.stat_["samples"] == self.max_samples_
            ):
                num_samples = (
                    self.stat_["acc_micro_batches"] * self.num_samples_per_micro_batch_
                )
                throughput = num_samples / self.stat_["acc_elapsed_time"]
                self.stat_["throughput"] = throughput
                latency = self.stat_["acc_elapsed_time"] / self.stat_["acc_batches"]
                self.stat_["latency"] = latency

                for key in self.keys_:
                    value = self.stat_[key] / self.stat_["acc_micro_batches"]
                    self.stat_[key] = value

                self.print_fn_()

                self.step_stat()

            # metric_time = self.timer_.step()
            # print("==>", metric_time)

        return callback
