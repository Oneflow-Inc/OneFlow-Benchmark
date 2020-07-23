from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow
from optimizer_util import gen_model_update_conf


def _default_config(args):
    config = flow.function_config()
    config.default_distribute_strategy(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    return config


def get_train_config(args):
    train_config = _default_config(args)
    train_config.train.primary_lr(args.learning_rate)


    train_config.prune_parallel_cast_ops(True)
    train_config.train.model_update_conf(gen_model_update_conf(args))
    train_config.enable_inplace(True)
    return train_config


def get_val_config(args):
    return _default_config(args)
