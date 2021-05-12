import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import oneflow as flow

from oneflow_gpt.config import get_args
from oneflow_gpt import distribute
from oneflow_gpt.data import GPTDataLoader, get_train_val_test_num_samples
from oneflow_gpt.model import GPTModel, ParallelSparseSoftmaxCrossEntropyLoss
from oneflow_gpt.optimizer import make_optimizer
from oneflow_gpt.snapshot import Snapshot
from oneflow_gpt.util import Metric
from oneflow_gpt.third_party.data.gpt_dataset import build_train_valid_test_datasets


def _init_env(args):
    if args.num_nodes > 1:
        if args.num_nodes > len(args.node_ips):
            raise ValueError(
                f"num_nodes {args.num_nodes} greater than"
                " length of node ips {args.node_ips}"
            )

        flow.env.ctrl_port(args.ctrl_port)
        nodes = []
        for ip in args.node_ips[: args.num_nodes]:
            nodes.append({"addr": ip})

        flow.env.machine(nodes)

    flow.env.log_dir(args.log)


def _init_config(args):
    flow.config.gpu_device_num(args.num_gpus_per_node)
    if args.tensor_model_parallel_size > 1:
        if hasattr(flow.config, "nccl_use_compute_stream"):
            flow.config.nccl_use_compute_stream(True)
        else:
            print(
                "WARNING: This version of OneFlow dose not support placing nccl on compute stream"
                " please try other version."
            )

    if args.use_rdma:
        flow.config.use_rdma(True)

    flow.config.enable_legacy_model_io()
    flow.config.enable_model_io_v2(True)


def _make_func_config(args):
    func_cfg = flow.function_config()
    if args.fp16:
        func_cfg.enable_auto_mixed_precision(True)
    func_cfg.prune_parallel_cast_ops(True)
    func_cfg.enable_fuse_add_to_output(True)
    func_cfg.enable_fuse_model_update_ops(True)
    func_cfg.enable_fuse_cast_scale(True)
    # turn on this flag when match ZeRO & DeepSpeed
    func_cfg.enable_non_distributed_optimizer(False)
    if args.num_accumulation_steps > 1:
        if hasattr(func_cfg.train, "num_gradient_accumulation_steps"):
            func_cfg.train.num_gradient_accumulation_steps(args.num_accumulation_steps)
        else:
            args.num_accumulation_steps = 1
            print(
                "WARNING: This version of OneFlow dose not support gradient accumulation"
                " please try newer version."
            )

    return func_cfg


def _make_gpt_train_func(args):
    model = GPTModel("model")
    loss = ParallelSparseSoftmaxCrossEntropyLoss()
    optimizer = make_optimizer(args)

    if args.use_external_dataset:

        @flow.global_function("train", _make_func_config(args))
        def train(
            x: flow.typing.Numpy.Placeholder(
                (args.global_batch_size, args.seq_length + 1), dtype=flow.int64
            )
        ):
            x = distribute.input_data_parallel_cast(x)
            with distribute.layer_placement_scope(0):
                data = flow.slice(x, begin=(None, 0), size=(None, args.seq_length))
            with distribute.layer_placement_scope(-1):
                labels = flow.slice(x, begin=(None, 1), size=(None, args.seq_length))

            logits = model(data)
            losses = loss(logits, labels)
            optimizer.minimize(losses)

            losses = distribute.output_parallel_cast(losses)
            return {"loss": losses}

    else:
        data_loader = GPTDataLoader("gpt_data_loader")

        @flow.global_function("train", _make_func_config(args))
        def train():
            data, labels = data_loader()
            logits = model(data)
            losses = loss(logits, labels)
            optimizer.minimize(losses)

            losses = distribute.output_parallel_cast(losses)
            return {"loss": losses}

    return train


def train():
    args = get_args()
    _init_env(args)
    _init_config(args)
    trainer = _make_gpt_train_func(args)
    snapshot = Snapshot(trainer.__name__)

    metric = Metric(
        print_steps=args.log_interval,
        num_samples_per_batch=args.micro_batch_size * args.data_parallel_size,
        max_samples=args.train_samples,
        keys=["loss"],
        print_format=args.metric_print_format,
    )

    if args.use_external_dataset:
        train_val_test_num_samples = get_train_val_test_num_samples(
            args.split, args.train_samples
        )
        train_ds, _, _ = build_train_valid_test_datasets(
            data_prefix=[args.dataset],
            data_impl="mmap",
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=0,
        )

    print("Training...")
    try:
        batch_size = args.micro_batch_size * args.num_accumulation_steps
        iteration = snapshot.iter
        while iteration < args.train_iters:
            if args.use_external_dataset:
                batch = [
                    train_ds[iteration * batch_size + i] for i in range(batch_size)
                ]
                data = np.stack(batch)
                trainer(data).async_get(metric.metric_cb())
            else:
                trainer().async_get(metric.metric_cb())

            snapshot.step()
            iteration = snapshot.iter

    except KeyboardInterrupt:
        print("interrupted")


if __name__ == "__main__":
    train()
