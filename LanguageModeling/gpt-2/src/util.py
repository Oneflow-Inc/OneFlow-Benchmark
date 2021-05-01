import os
import time
import math
from collections import OrderedDict
from datetime import datetime

import oneflow as flow


def init_env(args):
    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        flow.env.ctrl_port(args.ctrl_port)
        nodes = []
        for ip in args.node_ips[: args.num_nodes]:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)

    flow.env.log_dir(args.log_dir)


def init_config(args):
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.collective_boxing.nccl_fusion_reduce_scatter(True)
    flow.config.collective_boxing.nccl_fusion_all_gather(True)
    flow.config.collective_boxing.nccl_enable_mixed_fusion(True)
    # flow.config.enable_legacy_model_io(True)
    if args.nccl_use_compute_stream:
        flow.config.nccl_use_compute_stream(True)
    else:
        flow.config.nccl_use_compute_stream(False)
    if args.disable_group_boxing_by_dst_parallel:
        flow.config.disable_group_boxing_by_dst_parallel(True)
    else:
        flow.config.disable_group_boxing_by_dst_parallel(False)


def make_func_config(args):
    config = flow.function_config()
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    config.prune_parallel_cast_ops(True)
    config.enable_fuse_add_to_output(True)
    config.enable_fuse_model_update_ops(True)
    config.enable_fuse_cast_scale(True)
    # turn on the flag of none-distributed-optimizer
    if args.enable_non_distributed_optimizer:
        config.train.optimizer_placement_optimization_mode("distributed_split")
    return config


def make_optimizer(args):
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate])
    loss_scale_policy = None
    if args.use_fp16:
        flow.optimizer.loss_scale.dynamic_loss_scale(increment_period=20)

    if args.optimizer == "adam":
        optimizer = flow.optimizer.Adam(
            lr_scheduler, do_bias_correction=True, loss_scale_policy=loss_scale_policy
        )
    elif args.optimizer == "sgd":
        optimizer = flow.optimizer.SGD(lr_scheduler, momentum=0.0)
    elif args.optimizer == "adamw":
        optimizer = flow.optimizer.AdamW(
            lr_scheduler,
            do_bias_correction=True,
            loss_scale_policy=loss_scale_policy,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_eps,
            weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
            weight_decay=args.weight_decay,
            grad_clipping=flow.optimizer.grad_clipping.by_global_norm(args.clip_grad),
        )
    else:
        raise ValueError("Unsupported optimizer:", args.optimizer)
    return optimizer


def pad_vocab_size(vocab_size, alignment, num_devices, parallel_embedding):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    assert isinstance(alignment, int)
    if alignment == 0:
        return vocab_size

    if parallel_embedding:
        alignment *= num_devices

    padded_vocab_size = int(math.ceil(vocab_size / alignment)) * alignment
    print(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(
            vocab_size, padded_vocab_size - vocab_size, padded_vocab_size
        )
    )
    return padded_vocab_size


class Snapshot(object):
    def __init__(self, base_path=None, dir_prefix="model_save"):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_save_path_ = os.path.join(
            base_path or "", f"{dir_prefix}_{timestamp}"
        )

    def save(self, name):
        save_path = os.path.join(self.model_save_path_, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"Saving model to {save_path}")
        flow.checkpoint.save(save_path)


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        now = time.perf_counter()
        self.start_time = now
        self.last_split = now

    def split(self):
        now = time.perf_counter()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.perf_counter()

    def duration(self):
        return self.stop_time - self.start_time


class Metric(object):
    def __init__(
        self,
        desc="train",
        print_steps=-1,
        batch_size=256,
        keys=[],
        print_format="normal",
    ):
        r"""accumulate and calculate metric

        Args:
            desc: `str` general description of the metric to show
            print_steps: `Int` print metrics every nth steps
            batch_size: `Int` batch size per step
            keys: keys in callback outputs
        Returns:
        """
        self.desc = desc
        self.print_steps = print_steps
        assert batch_size > 0
        self.batch_size = batch_size

        assert isinstance(keys, (list, tuple))
        self.keys = keys
        self.metric_dict = OrderedDict()
        self.metric_dict["step"] = 0

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

        if print_format == "normal":
            self.print_fn = self.step_print
        elif print_format == "table":
            self.print_fn = self.step_print_by_table
        else:
            raise ValueError("print_format must be <normal|table>")

    def _clear(self):
        for key in self.keys:
            self.metric_dict[key] = 0.0
        self.metric_dict["throughput"] = 0.0
        self.num_samples = 0.0

    def update_and_save(self, key, value, step, **kwargs):
        self.metric_dict[key] = value

    def step_print(self):
        print(
            f"step={self.metric_dict['step']},"
            f"loss={self.metric_dict['loss']:.5f},"
            f"throughput={self.metric_dict['throughput']:.5f},"
            f"lantency={self.metric_dict['lantency']:.5f}"
        )

    def step_print_by_table(self):
        if self.metric_dict["step"] == self.print_steps:
            print(
                f"| {'step'.ljust(8)} "
                f"| {'loss'.ljust(10)} "
                f"| {'throughput'.ljust(10)} "
                f"| {'lantency'.ljust(10)} "
                "|"
            )
            print(f"| {'-' * 8} | {'-' * 10} | {'-' * 10} | {'-' * 10} |")

        print(
            f"| {self.metric_dict['step']:<8d} "
            f"| {self.metric_dict['loss']:<10.5f} "
            f"| {self.metric_dict['throughput']:<10.5f} "
            f"| {self.metric_dict['lantency']:<10.5f} "
            "|"
        )

    def metric_cb(self, step=0, **kwargs):
        def callback(outputs):
            if step == 0:
                self._clear()

            for key in self.keys:
                self.metric_dict[key] += outputs[key].sum()

            self.num_samples += self.batch_size

            if (step + 1) % self.print_steps == 0:
                self.metric_dict["step"] = step + 1
                for k, v in kwargs.items():
                    self.metric_dict[k] = v

                elapsed_time = self.timer.split()
                throughput = self.num_samples / elapsed_time
                self.update_and_save("throughput", throughput, step)
                lantency = elapsed_time / self.print_steps
                self.update_and_save("lantency", lantency, step)
                for key in self.keys:
                    value = self.metric_dict[key] / self.num_samples
                    self.update_and_save(key, value, step, **kwargs)

                self.print_fn()
                self._clear()

        return callback
