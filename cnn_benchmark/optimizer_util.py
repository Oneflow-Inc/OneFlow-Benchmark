from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


def add_optimizer_args(parser):
    group = parser.add_argument_group('optimizer parameters',
                                      'entire group applies only to optimizer parameters')
    group.add_argument("--optimizer", type=str, default="momentum-cosine-decay",
                       help="sgd, adam, momentum, momentum-cosine-decay")
    # group.add_argument("--weight_decay_rate", type=float, default=1.0/32768, help="weight decay")
    group.add_argument("--learning_rate", type=float, default=0.256)
    group.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    return parser


def get_optimizer(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    epoch_size = math.ceil(args.num_examples / train_batch_size)
    num_train_batches = epoch_size * args.num_epochs
    num_warmup_batches = epoch_size * args.warmup_epochs
    decay_batches = num_train_batches - num_warmup_batches
    optimizer_dict = {
        "sgd": {"naive_conf": {}},
        "adam": {"adam_conf": {"beta1": 0.9}},
        "momentum": {"momentum_conf": {"beta": 0.9}},
        "momentum-decay": {
            "momentum_conf": {"beta": 0.9},
            "learning_rate_decay": {
                "polynomial_conf": {"decay_batches": 300000, "end_learning_rate": 0.0001, },
            },
        },
        "momentum-cosine-decay": {
            "momentum_conf": {"beta": 0.875},
            "warmup_conf": {"linear_conf": {"warmup_batches": num_warmup_batches, "start_multiplier": 0}},
            "learning_rate_decay": {"cosine_conf": {"decay_batches": decay_batches}},
            # "weight_decay_conf": {
            #    "weight_decay_rate": args.weight_decay_rate,
            #    #"excludes": {"pattern": ['', '']},
            #    "includes": {"pattern": ['weight']},
            # }
        },
    }
    return optimizer_dict[args.optimizer]
