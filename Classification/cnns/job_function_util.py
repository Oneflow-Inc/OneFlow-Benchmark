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


def _default_config(args):
    config = flow.function_config()
    config.default_logical_view(flow.scope.consistent_view())
    config.default_data_type(flow.float)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    if args.use_xla:
        config.use_xla_jit(True)
    config.enable_fuse_add_to_output(True)
    return config


def get_train_config(args):
    train_config = _default_config(args)
    train_config.cudnn_conv_heuristic_search_algo(False)

    train_config.prune_parallel_cast_ops(True)
    train_config.enable_inplace(True)
    if args.num_nodes > 1:
        train_config.cudnn_conv_heuristic_search_algo(True)
    else:
        train_config.cudnn_conv_heuristic_search_algo(False)
    train_config.enable_fuse_model_update_ops(True)
    return train_config


def get_val_config(args):
    return _default_config(args)
