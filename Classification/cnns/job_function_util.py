"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
from optimizer_util import gen_model_update_conf


def _default_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    if args.use_xla:
        config.use_xla_jit(True)
    return config


def get_train_config(args):
    train_config = _default_config(args)
    train_config.train.primary_lr(args.learning_rate)
    train_config.cudnn_conv_heuristic_search_algo(False)


    train_config.prune_parallel_cast_ops(True)
    train_config.train.model_update_conf(gen_model_update_conf(args))
    train_config.enable_inplace(True)
    return train_config


def get_val_config(args):
    return _default_config(args)
