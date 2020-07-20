from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
from collections import OrderedDict

import oneflow as flow

from binclassiflication import TrainBert
import benchmark_util

from parse_args import parse_args
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import os

flow.config.gpu_device_num(3)
args = parse_args()


def _blob_conf(name, shape, dtype=flow.int32):
    return flow.data.BlobConf(
        name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec()
    )


def BertDecoder(
    data_dir, batch_size, data_part_num, seq_length, part_name_prefix
):
    config_ordered_dict = OrderedDict()
    config_ordered_dict['input_ids'] = seq_length
    config_ordered_dict['input_mask'] = seq_length
    config_ordered_dict['segment_ids'] = seq_length
    config_ordered_dict['label_ids'] = 1
    config_ordered_dict['is_real_example'] = 1

    blob_confs = []
    for k, v in config_ordered_dict.items():
        blob_confs.append(_blob_conf(
            k, [v], flow.float if k == 'masked_lm_weights' else flow.int32))

    decoders = flow.data.decode_ofrecord(
        data_dir,
        blob_confs,
        batch_size=batch_size,
        name="decode",
        data_part_num=1,
        part_name_prefix=part_name_prefix,
        part_name_suffix_length=-1,
    )

    ret = {}
    for i, k in enumerate(config_ordered_dict):
        ret[k] = decoders[i]
    return ret


def BuildBert(
    batch_size,
    data_part_num,
    data_dir,
    part_name_prefix,
    seq_length=128,
    max_position_embeddings=512,
    num_hidden_layers=12,
    num_attention_heads=12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    vocab_size=30522,
    type_vocab_size=2,
):
    hidden_size = 64 * num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    decoders = BertDecoder(
        data_dir, batch_size, data_part_num, seq_length, part_name_prefix
    )

    input_ids = decoders['input_ids']
    input_mask = decoders['input_mask']
    # note: segment_ids = token_type_ids
    token_type_ids = decoders['segment_ids']
    label_ids = decoders['label_ids']
    is_real_example = decoders['is_real_example']

    loss, logits = TrainBert(
        input_ids,
        input_mask,
        token_type_ids,
        label_ids,
        vocab_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=0.02,
    )
    return loss, logits, label_ids


# _BERT_MODEL_UPDATE_CONF = dict(
#     learning_rate_decay=dict(
#         polynomial_conf=dict(
#             decay_batches=args.iter_num if args.lr_decay_num_same_as_iter_num else args.lr_decay_num,
#             end_learning_rate=0.0,)
#     ),
#     warmup_conf=dict(linear_conf=dict(warmup_batches=args.warmup_batches, start_multiplier=0,)),
#     clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0,)),
#     adam_conf=dict(epsilon=1e-6),
# )

_BERT_MODEL_UPDATE_CONF = dict(
    learning_rate_decay=dict(
        polynomial_conf=dict(
            decay_batches=args.iter_num if args.lr_decay_num_same_as_iter_num else args.lr_decay_num,
            end_learning_rate=0,)
    ),
    warmup_conf=dict(linear_conf=dict(
        warmup_batches=args.warmup_batches, start_multiplier=0,)),
    clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0,)),
    adam_conf=dict(epsilon=1e-6),
    weight_decay_conf=dict(weight_decay_rate=args.weight_decay_rate,
                           #excludes=dict(pattern=['bias', 'LayerNorm', 'layer_norm'])
                           )
)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.train.primary_lr(args.learning_rate)
func_config.default_data_type(flow.float)
func_config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
# func_config.disable_all_reduce_sequence(True)
# func_config.all_reduce_group_min_mbyte(8)
# func_config.all_reduce_group_num(128)

# if args.weight_l2:
#     func_config.train.weight_l2(args.weight_l2)

flow.config.gpu_device_num(args.gpu_num_per_node)
if args.enable_auto_mixed_precision:
    func_config.enable_auto_mixed_precision()

val_func_config = flow.function_config()
val_func_config.default_distribute_strategy(
    flow.distribute.consistent_strategy())
val_func_config.default_data_type(flow.float)

#
@flow.global_function(func_config)
def BertGlueFinetuneJob():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    loss, logits, _ = BuildBert(
        batch_size,
        args.data_part_num,
        args.data_dir,
        'train.of_record-',
        seq_length=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        vocab_size=args.vocab_size,
        type_vocab_size=args.type_vocab_size,
    )
    flow.losses.add_loss(loss)
    return loss


@flow.global_function(val_func_config)
def BertGlueEvalTrainJob():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    #8551 or 1042
    _, logits, label_ids = BuildBert(
        batch_size,
        args.data_part_num,
        args.data_dir,
        'train.of_record-',
        seq_length=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        vocab_size=args.vocab_size,
        type_vocab_size=args.type_vocab_size,
    )

    return logits, label_ids


@flow.global_function(val_func_config)
def BertGlueEvalValJob():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    #8551 or 1042
    _, logits, label_ids = BuildBert(
        batch_size,
        args.data_part_num,
        args.val_data_dir,
        'eval.of_record-',
        seq_length=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        vocab_size=args.vocab_size,
        type_vocab_size=args.type_vocab_size,
    )

    return logits, label_ids

# @flow.global_function(val_func_config)
# def BertGlueEvalTestJob():
#     total_device_num = args.node_num * args.gpu_num_per_node
#     batch_size = total_device_num * args.batch_size_per_device
#     #8551 or 1042
#     _,logits,label_ids = BuildBert(
#         batch_size,
#         args.data_part_num,
#         args.test_data_dir,
#         'predict.of_record-',
#         seq_length=args.seq_length,
#         max_position_embeddings=args.max_position_embeddings,
#         num_hidden_layers=args.num_hidden_layers,
#         num_attention_heads=args.num_attention_heads,
#         hidden_dropout_prob=args.hidden_dropout_prob,
#         attention_probs_dropout_prob=args.attention_probs_dropout_prob,
#         vocab_size=args.vocab_size,
#         type_vocab_size=args.type_vocab_size,
#     )

#     return logits,label_ids


def EvalModel(step):
    if args.task == 'CoLA':
        len_train_data = 8551
    elif args.task == 'MRPC':
        len_train_data = 3668
    train_labels = []
    train_predictions = []
    for index in range(len_train_data//(args.node_num * args.gpu_num_per_node * args.batch_size_per_device)):
        train_logits, train_label = BertGlueEvalTrainJob().get()
        train_predictions.extend(list(train_logits.numpy().argmax(axis=1)))
        train_labels.extend(list(train_label))

    if args.task == 'CoLA':
        print('train mcc', matthews_corrcoef(train_labels, train_predictions))
    elif args.task == 'MRPC':
        train_acc = np.mean(np.array(train_predictions)
                            == np.array(train_labels))
        print('train acc', train_acc)

    if args.task == 'CoLA':
        len_val_data = 1042
    elif args.task == 'MRPC':
        len_val_data = 408

    val_labels = []
    val_predictions = []
    for index in range(len_val_data//(args.node_num * args.gpu_num_per_node * args.batch_size_per_device)):
        val_logits, val_label = BertGlueEvalValJob().get()
        val_predictions.extend(list(val_logits.numpy().argmax(axis=1)))
        val_labels.extend(list(val_label))

    if args.task == 'CoLA':
        print('val mcc', matthews_corrcoef(val_labels, val_predictions))
    elif args.task == 'MRPC':
        val_acc = np.mean(np.array(val_predictions) == np.array(val_labels))
        print('val acc', val_acc)

    # print('writing predictions to ./predictions.csv')
    # pd.DataFrame({'predictions': val_predictions, 'labels': val_labels}).to_csv('predictions_{0}.csv'.format(step), index=False)
    # len_test_data = 1064
    # test_labels=[]
    # test_predictions=[]
    # for index in range(len_test_data//(args.node_num * args.gpu_num_per_node * args.batch_size_per_device)):
    #     test_logits,test_label = BertGlueEvalTestJob().get()
    #     test_predictions.extend(list(test_logits.ndarray().argmax(axis=1)))
    #     test_labels.extend(list(test_label))

    # test_acc = np.mean(np.array(test_predictions) == np.array(test_labels))
    # print('test acc',test_acc)
    # print('test mcc',matthews_corrcoef(test_labels,test_predictions))
    #print('writing predictions to ./predictions.csv')
    #pd.DataFrame({'predictions': val_predictions, 'labels': val_labels}).to_csv('predictions_{0}.csv'.format(step), index=False)


def main():

    print("=".ljust(66, "="))
    print(
        "Running bert: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.gpu_num_per_node, args.node_num
        )
    )
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(
        str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

    flow.env.log_dir(args.log_dir)

    if args.node_num > 1:
        print('| Node number: {}'.format(args.node_num))
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)

    check_point = flow.train.CheckPoint()
    if args.model_load_dir:
        assert os.path.isdir(args.model_load_dir)
        check_point.load(args.model_load_dir)
        print("Restoring model from {}.".format(args.model_load_dir))
    else:
        check_point.init()
        # save_and_break=False
        if args.save_and_break:
            print("| Init model on demand")
            if os.path.exists(args.model_save_dir):
                import shutil
                shutil.rmtree(args.model_save_dir)
            print("save model just after init and exit")
            check_point.save(args.model_save_dir)
            import sys
            sys.exit()

    total_batch_size = (
        args.node_num * args.gpu_num_per_node * args.batch_size_per_device
    )
    print('batch size:', total_batch_size)
    speedometer = benchmark_util.BERTSpeedometer()
    start_time = time.time()

    for step in range(args.skip_iter_num + args.iter_num):
        cb = speedometer.speedometer_cb(
            step,
            start_time,
            total_batch_size,
            args.skip_iter_num,
            args.iter_num,
            args.loss_print_every_n_iter,
        )

        # BertGlueFintuneJob().get()
        BertGlueFinetuneJob().async_get(cb)
        # BertGlueFinetuneJob().get()
        # cb(BertGlueFinetuneJob().get())

        if step % args.loss_print_every_n_iter == 0:
            print('start caculate acc')
            EvalModel(step)

        if (step + 1) % args.model_save_every_n_iter == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            snapshot_save_path = os.path.join(
                args.model_save_dir, "snapshot_%d" % (step + 1)
            )
            print("Saving model to {}.".format(snapshot_save_path))
            check_point.save(snapshot_save_path)

    if args.save_last_snapshot:
        snapshot_save_path = args.model_save_dir
        if os.path.exists(args.model_save_dir):
            import shutil
            shutil.rmtree(args.model_save_dir)
        print("Saving model to {}.".format(snapshot_save_path))
        check_point.save(snapshot_save_path)


if __name__ == "__main__":
    main()
