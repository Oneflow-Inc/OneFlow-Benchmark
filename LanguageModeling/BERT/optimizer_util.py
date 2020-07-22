from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow

def gen_model_update_conf(args):
    _BERT_MODEL_UPDATE_CONF = dict(
        learning_rate_decay=dict(
            polynomial_conf=dict(
                decay_batches=args.iter_num,
                end_learning_rate=0.0,
            )
        ),
        warmup_conf=dict(
            linear_conf=dict(warmup_batches=args.warmup_batches, start_multiplier=0,)
        ),
        clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0,)),
        adam_conf=dict(epsilon=1e-6),
        weight_decay_conf=dict(
            weight_decay_rate=args.weight_decay_rate,
            excludes=dict(pattern=["bias", "LayerNorm", "layer_norm"]),
        ),
    )
    
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.default_distribute_strategy(flow.scope.consistent_view())
    config.train.primary_lr(args.learning_rate)
    config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
    
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    
    return config