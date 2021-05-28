import math
import argparse

_GLOBAL_ARGS = None


def get_args(extra_args_provider=None):
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args(extra_args_provider)

    return _GLOBAL_ARGS


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(
        description="OneFlow GPT Arguments", allow_abbrev=False
    )
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_misc_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    _check_model_size(args)
    _check_parallel_size(args)
    _check_batch_size(args)
    _check_train_iters(args)
    _check_lr_decay_and_warmup(args)

    args.padded_vocab_size = _pad_vocab_size(
        args.vocab_size,
        args.make_vocab_size_divisible_by,
        args.tensor_model_parallel_size,
    )

    _print_args(args)
    return args


def _str_list(x):
    return x.split(",")


def _int_list(x):
    return list(map(int, x.split(",")))


def _float_list(x):
    return list(map(float, x.split(",")))


def _str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def _check_model_size(args):
    assert isinstance(args.num_layers, int)
    assert isinstance(args.hidden_size, int)
    assert isinstance(args.num_attention_heads, int)
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError(
            f"hidden size {args.hidden_size} must be divisible by"
            f" number of attention heads {args.num_attention_heads}"
        )

    if args.num_attention_heads % args.tensor_model_parallel_size != 0:
        raise ValueError(
            f"number of attention heads {args.num_attention_heads} must be divisible by"
            f" tensor model parallel size {args.tensor_model_parallel_size}"
        )

    if args.num_layers % args.pipeline_model_parallel_size != 0:
        raise ValueError(
            f"number of layers {args.num_layers} must be divisible by"
            f" pipeline model parallel size {args.pipeline_model_parallel_size}"
        )


def _check_parallel_size(args):
    if len(args.node_ips) < args.num_nodes:
        raise ValueError(
            f"number of node ips {args.node_ips} less than num_nodes {args.num_nodes}"
        )

    world_size = args.num_gpus_per_node * args.num_nodes
    model_parallel_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size {world_size} must be divisible by model_parallel_size {model_parallel_size}"
        )

    args.data_parallel_size = world_size // model_parallel_size


def _check_batch_size(args):
    if args.micro_batch_size is not None and args.global_batch_size is not None:
        if args.num_accumulation_steps is None:
            if (
                args.global_batch_size
                % (args.micro_batch_size * args.data_parallel_size)
                != 0
            ):
                raise ValueError(
                    f"global_batch_size {args.global_batch_size} must be divisible by "
                    f"micro_batch_size * data_parallel_size ({args.micro_batch_size} * {args.data_parallel_size})"
                )

            args.num_accumulation_steps = args.global_batch_size // (
                args.micro_batch_size * args.data_parallel_size
            )
        else:
            if (
                args.global_batch_size
                != args.micro_batch_size
                * args.data_parallel_size
                * args.num_accumulation_steps
            ):
                raise ValueError(
                    f"global_batch_size {args.global_batch_size} must equal"
                    " micro_batch_size * data_parallel_size * num_accumulation_steps"
                    f" ({args.micro_batch_size} * {args.data_parallel_size} * {args.num_accumulation_steps})"
                )
    elif args.micro_batch_size is not None and args.global_batch_size is None:
        if args.num_accumulation_steps is None:
            args.num_accumulation_steps = 1

        args.global_batch_size = (
            args.micro_batch_size
            * args.data_parallel_size
            * args.num_accumulation_steps
        )
    elif args.micro_batch_size is None and args.global_batch_size is not None:
        if args.num_accumulation_steps is None:
            args.num_accumulation_steps = 1

        if (
            args.global_batch_size
            % (args.data_parallel_size * args.num_accumulation_steps)
            != 0
        ):
            raise ValueError(
                f"global_batch_size {args.global_batch_size} must be divisible by "
                "data_parallel_size * num_accumulation_steps "
                f"({args.data_parallel_size} * {args.num_accumulation_steps})"
            )

        args.micro_batch_size = args.global_batch_size // (
            args.data_parallel_size * args.num_accumulation_steps
        )
    else:
        raise ValueError("micro_batch_size and global_batch_size must be set either")

    assert args.num_accumulation_steps is not None
    if args.num_accumulation_steps > 1 and args.use_external_dataset:
        raise ValueError(
            "num_accumulation_steps couldn't be greater than 1 when use external dataset"
        )


def _check_train_iters(args):
    if args.train_iters is None and args.train_samples is None:
        return

    if args.train_iters is None:
        if args.train_samples % args.global_batch_size != 0:
            raise ValueError(
                f"train_samples {args.train_samples} must be divisible by "
                f"global_batch_size {args.global_batch_size}"
            )

        args.train_iters = args.train_samples // args.global_batch_size

    if args.train_samples is None:
        args.train_samples = args.train_iters * args.global_batch_size


def _check_lr_decay_and_warmup(args):
    if args.lr_decay_style == "cosine" and args.lr_decay_iters is None:
        raise ValueError(
            f"lr_decay_iters must be set when lr_decay_style is {args.lr_decay_style}"
        )

    if (
        args.lr_warmup_iters is None
        and args.lr_warmup_fraction is not None
        and args.lr_decay_iters is not None
    ):
        args.lr_warmup_iters = int(args.lr_warmup_fraction * args.lr_decay_iters)


def _pad_vocab_size(vocab_size, alignment, tensor_model_parallel_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    assert isinstance(alignment, int)
    if alignment == 0:
        return vocab_size

    alignment *= tensor_model_parallel_size

    padded_vocab_size = int(math.ceil(vocab_size / alignment)) * alignment
    print(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(
            vocab_size, padded_vocab_size - vocab_size, padded_vocab_size
        )
    )
    return padded_vocab_size


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title="network size")

    group.add_argument(
        "--num-layers", type=int, default=None, help="Number of transformer layers."
    )
    group.add_argument(
        "--hidden-size", type=int, default=None, help="Tansformer hidden size."
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=None,
        help="Number of transformer attention heads.",
    )
    group.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=128,
        help="Pad the vocab size to be divisible by this value."
        "This is added for computational efficieny reasons.",
    )

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title="regularization")

    group.add_argument(
        "--attention-dropout",
        type=float,
        default=0.1,
        help="Post attention dropout probability.",
    )
    group.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for hidden state transformer.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
    )
    group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="First coefficient for computing running averages of"
        "gradient and its square",
    )
    group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Second coefficient for computing running averages of"
        "gradient and its square",
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-08,
        help="Term added to the denominator to improve" "numerical stability",
    )

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Batch size per model instance (local batch size). "
        "Global batch size is local batch size times data "
        "parallel size times number of micro batches.",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Training batch size. If set, it should be a "
        "multiple of micro-batch-size times data-parallel-size. "
        "If this value is None, then "
        "use micro-batch-size * data-parallel-size as the "
        "global batch size. This choice will result in 1 for "
        "number of micro-batches.",
    )
    parser.add_argument(
        "--num-accumulation-steps",
        type=int,
        default=None,
        help="Number of accumulation micro steps before gradient update, "
        "Global batch size = num_accumulation_steps * batch_size",
    )
    group.add_argument(
        "--checkpoint-activations",
        action="store_true",
        help="Checkpoint activation to allow for training "
        "with larger models, sequences, and batch sizes.",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Total number of iterations to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )
    group.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Total number of samples to train over all "
        "training runs. Note that either train-iters or "
        "train-samples should be provided.",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer. <sgd|adam|adamw>.",
    )
    group.add_argument(
        "--no-scale-tril-softmax-dropout-fusion",
        action="store_false",
        help="Disable upper triangular version of fused scale, mask, softmax fusion kernel.",
        dest="scale_tril_softmax_dropout_fusion",
    )
    group.add_argument(
        "--no-bias-gelu-fusion",
        action="store_false",
        help="Disable bias and gelu fusion.",
        dest="bias_gelu_fusion",
    )
    group.add_argument(
        "--no-bias-dropout-fusion",
        action="store_false",
        help="Disable bias and dropout fusion.",
        dest="bias_dropout_fusion",
    )
    group.add_argument(
        "--multihead-attention-fusion",
        action="store_true",
        help="open transformer layer profiler",
    )
    group.add_argument("--log", type=str, default="./output", help="log directory")
    group.add_argument(
        "--log-interval", type=int, default=100, help="Report loss and timing interval."
    )
    group.add_argument(
        "--metric-print-format",
        type=str,
        default="table",
        choices=["normal", "table"],
        help="metric print format <normal|table>",
    )

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")

    group.add_argument(
        "--init-method-std",
        type=float,
        default=0.02,
        help="Standard deviation of the zero mean normal "
        "distribution used for weight initialization.",
    )

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title="learning rate")

    group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate. Depending on decay style "
        "and initial warmup, the learing rate at each "
        "iteration would be different.",
        dest="lr",
    )
    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine"],
        help="Learning rate decay function.",
    )
    group.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over,"
        " If None defaults to `--train-iters`",
    )
    group.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=None,
        help="fraction of lr-warmup-(iters/samples) to use " "for warmup (as a float)",
    )
    group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=None,
        help="number of iterations to linearly warmup " "learning rate over.",
    )
    group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum value for learning rate. The scheduler"
        "clip values below this threshold.",
    )

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title="checkpointing")

    group.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )
    group.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Number of iterations between checkpoint saves.",
    )
    group.add_argument(
        "--load",
        type=str,
        default=None,
        help="Directory containing a model checkpoint.",
    )
    group.add_argument(
        "--save-last",
        action="store_true",
        default=False,
        help="save model snapshot for last iteration",
    )
    group.add_argument(
        "--save-init",
        action="store_true",
        default=False,
        help="save model snapshot for inited",
    )

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title="mixed precision")

    group.add_argument("--fp16", action="store_true", help="Run model in fp16 mode.")
    group.add_argument(
        "--loss-scale",
        type=float,
        default=None,
        help="Static loss scaling, positive power of 2 "
        "values can improve fp16 convergence. If None, dynamic"
        "loss scaling is used.",
    )
    group.add_argument(
        "--initial-loss-scale",
        type=float,
        default=2 ** 32,
        help="Initial loss-scale for dynamic loss scaling.",
    )
    group.add_argument(
        "--loss-scale-window",
        type=float,
        default=1000,
        help="Window over which to raise/lower dynamic scale.",
    )
    group.add_argument(
        "--no-query-key-layer-scaling",
        action="store_false",
        help="Do not scale Q * K^T by 1 / layer-number.",
        dest="apply_query_key_layer_scaling",
    )

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title="distributed")

    group.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism.",
    )
    group.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    group.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=1,
        help="number of gpu devices per node/machine",
    )
    group.add_argument(
        "--num-nodes", type=int, default=1, help="node/machine number for training"
    )
    group.add_argument(
        "--node-ips",
        type=_str_list,
        default=[],
        help='nodes ip list for training, devided by ",", length >= num_nodes',
    )
    group.add_argument(
        "--ctrl-port", type=int, default=50051, help="ctrl_port for multinode job"
    )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title="validation")
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title="data and dataloader")

    group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path join the data index file and binary file prefix.",
    )
    group.add_argument(
        "--split",
        type=_int_list,
        default=[969, 30, 1],
        help="Comma-separated list of proportions for training,"
        " validation, and test split. For example the split "
        "`90,5,5` will use 90%% of data for training, 5%% for "
        "validation and 5%% for test.",
    )
    group.add_argument(
        "--seed", type=int, default=12345, help="Random seed used for data gen."
    )
    group.add_argument("--vocab-size", type=int, default=50257, help="Vocab size.")
    group.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length to process.",
    )
    group.add_argument(
        "--use-external-dataset",
        action="store_true",
        help="Use external megatron dataset.",
    )

    return parser


def _add_misc_args(parser):
    group = parser.add_argument_group(title="misc")
    group.add_argument(
        "--profile-transformer-layer",
        action="store_true",
        help="open transformer layer profiler",
    )
    group.add_argument(
        "--use-rdma",
        action="store_true",
        help="Use rdma.",
    )
    return parser


if __name__ == "__main__":
    get_args()
