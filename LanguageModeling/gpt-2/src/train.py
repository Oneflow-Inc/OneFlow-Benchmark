import os
import sys
import json

import util
import config as configs


def set_python_path():
    absolute_path = os.path.join(os.getcwd(), __file__)
    gpt2_path = os.path.dirname(os.path.dirname(absolute_path))
    sys.path.insert(0, gpt2_path)


set_python_path()

from third_party.load_dataset import load_dataset, Sampler
from third_party import encoder

from model import GPT2
import oneflow.typing as tp
import oneflow as flow


def make_gpt2_train_func(args):
    @flow.global_function("train", util.GetFunctionConfig(args))
    def gpt2_func(
        x: flow.typing.Numpy.Placeholder(
            (args.batch_size, args.seq_len), dtype=flow.int64
        )
    ):
        outputs = {}
        gpt2 = GPT2(args)
        outputs = gpt2.forward(X)
        loss = gpt2.loss(x, outputs["logits"])
        outputs["loss"] = loss
        optimizer = util.make_optimizer(args)
        optimizer.minimize(loss)
        return {"loss": loss}

    return gpt2_func


def main():
    args = configs.get_args()
    # batch_size = args.batch_size_per_device * args.gpu_num_per_node * args.num_nodes
    args.batch_size = args.total_batch_size

    util.InitNodes(args)
    flow.env.log_dir(args.log_dir)
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.collective_boxing.nccl_fusion_reduce_scatter(True)
    flow.config.collective_boxing.nccl_fusion_all_gather(True)
    flow.config.collective_boxing.nccl_enable_mixed_fusion(True)

    gpt2_trainer = make_gpt2_train_func(args)
    snapshot = util.Snapshot(args.model_save_dir, args.model_load_dir)

    print("Loading dataset...")
    enc = encoder.get_encoder(args)
    chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
    data_sampler = Sampler(chunks, seed=1)
    print("dataset has", data_sampler.total_size, "tokens")

    metric = util.Metric(
        desc="train",
        print_steps=args.loss_print_every_n_iter,
        batch_size=args.batch_size,
        keys=["loss"],
    )
    print("Training...")
    try:
        for iter in range(args.iter_num):
            b = data_sampler.sample_batch(args.batch_size, args.seq_len)
            gpt2_trainer(b).async_get(metric.metric_cb(iter))
    except KeyboardInterrupt:
        # snapshot.save("last_snapshot")
        print("interrupted")


if __name__ == "__main__":
    main()
