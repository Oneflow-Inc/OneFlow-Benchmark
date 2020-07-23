from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from datetime import datetime

import config as configs
import oneflow as flow

from squad import SQuAD
from util import Snapshot, Summary, InitNodes, Metric
from optimizer_util import gen_model_update_conf

parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)


def SquadDecoder(data_dir, batch_size, data_part_num, seq_length, is_train=True):
    ofrecord = flow.data.ofrecord_reader(data_dir,
                                         batch_size=batch_size,
                                         data_part_num=data_part_num,
                                         random_shuffle = True,
                                         shuffle_after_epoch=True)
    blob_confs = {}
    def _blob_conf(name, shape, dtype=flow.int32):
        blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

    _blob_conf("input_ids", [seq_length])
    _blob_conf("input_mask", [seq_length])
    _blob_conf("segment_ids", [seq_length])
    if is_train:
        _blob_conf("start_positions", [1])
        _blob_conf("end_positions", [1])
    else:
        _blob_conf("unique_ids", [1])

    return blob_confs


@flow.global_function(gen_model_update_conf(args))
def SquadFinetuneJob():
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    total_device_num = args.num_nodes * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    decoders = SquadDecoder(args.data_dir, batch_size, args.data_part_num, args.seq_length)

    start_logits, end_logits = SQuAD(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
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
        initializer_range=0.02,
    )

    def _ComputeLoss(logits, positions):
        logits = flow.reshape(logits, [-1, args.seq_length])
        probs = flow.nn.softmax(logits)
        pre_example_loss = flow.nn.sparse_cross_entropy(labels=positions, prediction=probs)
        return pre_example_loss

    start_loss = _ComputeLoss(start_logits, decoders['start_positions'])
    end_loss = _ComputeLoss(end_logits, decoders['end_positions'])

    total_loss = 0.5*(start_loss + end_loss)
    flow.losses.add_loss(total_loss)
    return {'total_loss': total_loss}


def main():
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    InitNodes(args)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    total_batch_size = (
        args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
    )

    summary = Summary(args.log_dir, args)
    metric = Metric(desc='train', print_steps=args.loss_print_every_n_iter, summary=summary, 
                    batch_size=total_batch_size, keys=['total_loss'])

    for step in range(args.iter_num):
        SquadFinetuneJob().async_get(metric.metric_cb(step))

        if (step + 1) % args.model_save_every_n_iter == 0:
            snapshot.save("snapshot_%d" % (step + 1))

    if args.save_last_snapshot:
        snapshot.save("last_snapshot")


if __name__ == "__main__":
    main()
