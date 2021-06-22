import os
import time
import numpy as np


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
        start_step,
        max_step,
        num_samples_per_batch,
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
        self.max_step_ = max_step
        self.num_samples_per_batch_ = num_samples_per_batch

        self.nvidia_smi_report_step_ = nvidia_smi_report_step
        self.nvidia_smi_report_file_ = nvidia_smi_report_file

        self.step_ = start_step
        self.micro_batches_ = 0
        self.samples_ = 0
        self.throughput_ = 0.0
        self.latency_ = 0.0
        self.timestamp_ = 0.0

        self.kv_store_ = dict()
        if keys is None:
            self.keys_ = []
        else:
            self.keys_ = list(keys)

        for key in self.keys_:
            self.kv_store_[key] = 0.0

        # need reset after every print
        self.acc_elapsed_time_ = 0.0
        self.acc_micro_batches_ = 0
        self.acc_samples_ = 0

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
            f"step={self.step_},"
            f"micro_batches={self.micro_batches_},"
            f"samples={self.samples_},"
            f"throughput={self.throughput_:.5f},"
            f"latency={self.latency_:.5f},"
        )
        for key in self.keys_:
            record += f"{key}={self.kv_store_[key]:.5f},"

        print(record)

    def step_print_by_table(self):
        title = (
            f"| {'step'.ljust(8)} "
            f"| {'micro_batches'.ljust(15)} "
            f"| {'samples'.ljust(15)} "
            f"| {'throughput'.ljust(10)} "
            f"| {'latency'.ljust(10)} "
        )
        sep = f"| {'-' * 8} | {'-' * 15} | {'-' * 15} | {'-' * 10} | {'-' * 10} "

        record = (
            f"| {self.step_:<8d} "
            f"| {self.micro_batches_:<15d} "
            f"| {self.samples_:<15d} "
            f"| {self.throughput_:<10.5f} "
            f"| {self.latency_:<10.5f} "
        )

        for key in self.keys_:
            title += f"| {key.ljust(10)} "
            sep += f"| {'-' * 10} "
            record += f"| {self.kv_store_[key]:<10.5f} "

        title += "|"
        sep += "|"
        record += "|"

        if not self.print_title_:
            print(title)
            print(sep)
            self.print_title_ = True

        print(record)

    def metric_cb(self):
        def callback(outputs):
            elapsed_time = self.timer_.step()
            self.timestamp_ = self.timer_.cur_step()
            self.acc_elapsed_time_ += elapsed_time

            micro_batches = None
            for key in self.keys_:
                output = outputs[key].numpy()
                assert isinstance(output, np.ndarray)
                if micro_batches is None:
                    micro_batches = output.shape[0]
                else:
                    assert micro_batches == output.shape[0]
                self.kv_store_[key] += output.sum()

            self.step_ += 1
            self.acc_micro_batches_ += micro_batches
            self.acc_samples_ += micro_batches * self.num_samples_per_batch_

            if self.step_ == self.nvidia_smi_report_step_:
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
                if self.nvidia_smi_report_file_ is not None:
                    cmd += f" -f {self.nvidia_smi_report_file_}"
                os.system(cmd)
                self.print_title_ = False

            if self.step_ % self.print_steps_ == 0 or self.step_ == self.max_step_:
                self.throughput_ = self.acc_samples_ / self.acc_elapsed_time_
                self.latency_ = self.acc_elapsed_time_ / self.print_steps_

                for key in self.keys_:
                    value = self.kv_store_[key] / self.acc_micro_batches_
                    self.kv_store_[key] = value

                self.micro_batches_ += self.acc_micro_batches_
                self.samples_ += self.acc_samples_

                self.print_fn_()

                self.acc_elapsed_time_ = 0.0
                self.acc_micro_batches_ = 0
                self.acc_samples_ = 0

        return callback
