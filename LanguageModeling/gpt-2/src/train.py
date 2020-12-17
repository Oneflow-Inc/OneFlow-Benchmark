import oneflow as flow
from ..third_party.load_dataset import load_dataset, Sampler
from ..third_party.encoder import get_encoder

from . import config
from . import util
from .model import GPT2


def make_gpt2_train_func(args):
    @flow.global_function("train", util.make_func_config(args))
    def gpt2_func(
        x: flow.typing.Numpy.Placeholder(
            (args.batch_size, args.seq_len), dtype=flow.int64
        )
    ):
        if x.split_axis == 0:
            x = flow.parallel_cast(x, distribute=flow.distribute.broadcast())

        outputs = {}
        gpt2 = GPT2(args, name="model")
        outputs = gpt2.forward(x)
        loss = gpt2.loss(x, outputs["logits"], parallel_loss=args.parallel_loss)
        outputs["loss"] = loss
        optimizer = util.make_optimizer(args)
        optimizer.minimize(loss)
        return {"loss": loss}

    return gpt2_func


def train(args):
    util.init_env(args)
    util.init_config(args)
    gpt2_trainer = make_gpt2_train_func(args)
    snapshot = util.Snapshot(args.model_save_dir, args.model_load_dir)

    print("Loading dataset...")
    enc = get_encoder(args)
    chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
    data_sampler = Sampler(chunks, seed=1)
    print("dataset has", data_sampler.total_size, "tokens")

    metric = util.Metric(
        desc="train",
        print_steps=args.loss_print_every_n_iter,
        batch_size=args.batch_size,
        keys=["loss"],
        print_format=args.metric_print_format,
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
    args = config.get_args()
    train(args)
