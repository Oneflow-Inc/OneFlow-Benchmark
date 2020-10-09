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

import os
import argparse
from datetime import datetime

import config as configs
import oneflow as flow

from pretrain import PreTrain
from util import Snapshot, Summary, InitNodes, Metric, CreateOptimizer, GetFunctionConfig

parser = configs.get_parser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--data_part_num", type=int, default=32, help="data part number in dataset")
parser.add_argument("--iter_num", type=int, default=1144000, help="total iterations to run")
parser.add_argument("--batch_size_per_device", type=int, default=64)
args = parser.parse_args()
configs.print_args(args)

batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device


def BertDecoder(data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq):
    ofrecord = flow.data.ofrecord_reader(data_dir,
                                         batch_size=batch_size,
                                         data_part_num=data_part_num,
                                         random_shuffle = True,
                                         shuffle_after_epoch=True)
    blob_confs = {}
    def _blob_conf(name, shape, dtype=flow.int32):
        blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

    _blob_conf("input_ids", [seq_length])
    _blob_conf("next_sentence_labels", [1])
    _blob_conf("input_mask", [seq_length])
    _blob_conf("segment_ids", [seq_length])
    _blob_conf("masked_lm_ids", [max_predictions_per_seq])
    _blob_conf("masked_lm_positions", [max_predictions_per_seq])
    _blob_conf("masked_lm_weights", [max_predictions_per_seq], flow.float)
    return blob_confs

@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def PretrainJob():
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    if args.data_part_num == 1:
        with flow.scope.placement("cpu", "0:0"):
            decoders = BertDecoder(args.data_dir, batch_size, args.data_part_num, args.seq_length,
                                   args.max_predictions_per_seq)
    else:
        assert args.data_part_num > 1
        decoders = BertDecoder(args.data_dir, batch_size, args.data_part_num, args.seq_length,
                               args.max_predictions_per_seq)

    total_loss, mlm_loss, nsp_loss = PreTrain(
        decoders["input_ids"],
        decoders["input_mask"],
        decoders["segment_ids"],
        decoders["masked_lm_positions"],
        decoders["masked_lm_ids"],
        decoders["masked_lm_weights"],
        decoders["next_sentence_labels"],
        args.vocab_size,
        seq_length=args.seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        max_predictions_per_seq=args.max_predictions_per_seq,
        initializer_range=0.02,
        use_fp16=args.use_fp16,
    )
    opt = CreateOptimizer(args)
    opt.minimize(total_loss)
    return {'total_loss': total_loss, 'mlm_loss': mlm_loss, 'nsp_loss': nsp_loss}


def main():
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    InitNodes(args)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    summary = Summary(args.log_dir, args)
    metric = Metric(desc='train', print_steps=args.loss_print_every_n_iter, summary=summary, 
                    batch_size=batch_size, keys=['total_loss', 'mlm_loss', 'nsp_loss'])
    for step in range(args.iter_num):
        PretrainJob().async_get(metric.metric_cb(step))
        #PretrainJob().async_get(metric.metric_cb(step, epoch=3))
        if (step + 1) % args.model_save_every_n_iter == 0:
            snapshot.save("snapshot_%d" % (step + 1))

    if args.save_last_snapshot:
        snapshot.save("last_snapshot")


if __name__ == "__main__":
    main()
