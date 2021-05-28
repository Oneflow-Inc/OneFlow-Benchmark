import math
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from oneflow_gpt.model import GPTModel, ParallelSparseSoftmaxCrossEntropyLoss
from oneflow_gpt import util
from .datasets import build_dataset
import numpy as np
import oneflow as flow


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
    flow.config.collective_boxing.nccl_fusion_reduce_scatter(True)
    flow.config.collective_boxing.nccl_fusion_all_gather(True)
    flow.config.collective_boxing.nccl_enable_mixed_fusion(True)
    if args.tensor_model_parallel_size > 1:
        if hasattr(flow.config, "nccl_use_compute_stream"):
            flow.config.nccl_use_compute_stream(True)
        else:
            print(
                "WARNING: This version of OneFlow dose not support placing nccl on compute stream"
                " please try other version."
            )

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


def make_gpt_eval_func(args):
    @flow.global_function("predict", _make_func_config(args))
    def gpt_func(
        x: flow.typing.Numpy.Placeholder(
            (args.global_batch_size, args.seq_length), dtype=flow.int64
        )
    ):
        gpt = GPTModel("model")
        return gpt(x)

    return gpt_func


def process_batch(args, batch):
    """Process batch and produce inputs for the model."""

    loss_mask = batch["pad_mask"]
    tokens_ = batch["text"]
    labels = tokens_[:, 1:]
    tokens = tokens_[:, :-1]

    return tokens, labels, None, None, loss_mask


def forward_step(args, batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(args, batch)
    # Tell the model what our actual batch size will be
    # args.micro_batch_size = len(labels)

    # Forward model.

    # Forward pass through the model.
    logits = model(tokens).get()

    if eval_metric == "accuracy":
        bs, e = logits.numpy().shape
        outputs = np.argmax(
            logits.numpy().reshape(
                (args.micro_batch_size, int(bs / args.micro_batch_size), e)
            ),
            -1,
        )
        correct = (outputs == labels).astype(np.float32)
        correct[(1 - loss_mask).astype(np.bool_)] = 1
        correct = np.prod(correct, -1)
        return np.sum(correct)

    raise NotImplementedError(
        "forward method for evaluation metric {} "
        "is not implemented.".format(eval_metric)
    )

    return None


def evaluate(args, data_sets, model, eval_metric):
    """Evaluation."""
    total_output = 0.0

    # For all the batches in the dataset.
    for iteration in range(int(len(data_sets) / args.micro_batch_size)):
        text = [
            data_sets[iteration * args.micro_batch_size + i]["text"]
            for i in range(args.micro_batch_size)
        ]
        text = np.stack(text)
        pad_mask = [
            data_sets[iteration * args.micro_batch_size + i]["pad_mask"]
            for i in range(args.micro_batch_size)
        ]
        pad_mask = np.stack(pad_mask)
        if iteration % args.log_interval == 0:
            print("> working on iteration: {}".format(iteration))
        # Forward evaluation.
        output = forward_step(
            args, {"text": text, "pad_mask": pad_mask}, model, eval_metric
        )
        total_output += output

    return total_output


def evaluate_and_print_results(args, data_sets, model, eval_metric):
    """Evaluate and print results on screen."""
    # Evaluate and get results.
    output = evaluate(args, data_sets, model, eval_metric)

    string = " validation results on {} | ".format(args.task)
    if eval_metric == "accuracy":
        num_examples = (
            int(len(data_sets) / args.micro_batch_size) * args.micro_batch_size
        )
        acc = output / num_examples
        string += "number correct: {:.4E} | ".format(output)
        string += "total examples: {:.4E} | ".format(num_examples)
        string += "avg accuracy: {:.4E}".format(acc)
        print(string)
    else:
        raise NotImplementedError(
            "evaluation method for {} metric is not "
            "implemented yet.".format(eval_metric)
        )


def main(args):
    """Main program."""

    if args.task == "LAMBADA":
        eval_metric = "accuracy"
    else:
        raise NotImplementedError("{} task is not implemented.".format(args.task))

    # Set up model and load checkpoint.
    _init_env(args)
    _init_config(args)
    gpt_eval = make_gpt_eval_func(args)
    check_point = flow.train.CheckPoint()

    assert args.load is not None
    check_point.load(args.load)

    dataset = build_dataset(args)
    # Run evaluation.
    evaluate_and_print_results(args, dataset, gpt_eval, eval_metric)

    print("done :-)")
