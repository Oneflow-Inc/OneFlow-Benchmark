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
import math
import numpy as np

import oneflow as flow

from classifier import GlueBERT
from util import Snapshot, Summary, InitNodes, Metric, CreateOptimizer, GetFunctionConfig

import config as configs
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score

parser = configs.get_parser()
parser.add_argument("--task_name", type=str, default='CoLA')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument("--train_data_dir", type=str, default=None)
parser.add_argument("--train_data_prefix", type=str, default='train.of_record-')
parser.add_argument("--train_example_num", type=int, default=88614, 
                    help="example number in dataset")
parser.add_argument("--batch_size_per_device", type=int, default=32)
parser.add_argument("--train_data_part_num", type=int, default=1, 
                    help="data part number in dataset")
parser.add_argument("--eval_data_dir", type=str, default=None)
parser.add_argument("--eval_data_prefix", type=str, default='eval.of_record-')
parser.add_argument("--eval_example_num", type=int, default=10833, 
                    help="example number in dataset")
parser.add_argument("--eval_batch_size_per_device", type=int, default=64)
parser.add_argument("--eval_data_part_num", type=int, default=1, 
                    help="data part number in dataset")
args = parser.parse_args()

batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device

epoch_size = math.ceil(args.train_example_num / batch_size)
num_eval_steps = math.ceil(args.eval_example_num / eval_batch_size)
args.iter_num = epoch_size * args.num_epochs
configs.print_args(args)


def BertDecoder(
    data_dir, batch_size, data_part_num, seq_length, part_name_prefix, shuffle=True
):
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        blob_confs = {}
        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("label_ids", [1])
        _blob_conf("is_real_example", [1])

        return blob_confs


def BuildBert(
    batch_size,
    data_part_num,
    data_dir,
    part_name_prefix,
    shuffle=True
):
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    decoders = BertDecoder(
        data_dir, batch_size, data_part_num, args.seq_length, part_name_prefix, shuffle=shuffle
    )
    #is_real_example = decoders['is_real_example']

    loss, logits = GlueBERT(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
        decoders['label_ids'],
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
    return loss, logits, decoders['label_ids']


@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def BertGlueFinetuneJob():
    loss, logits, _ = BuildBert(
        batch_size,
        args.train_data_part_num,
        args.train_data_dir,
        args.train_data_prefix,
    )
    flow.losses.add_loss(loss)
    opt = CreateOptimizer(args)
    opt.minimize(loss)
    return {'loss': loss}


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalTrainJob():
    _, logits, label_ids = BuildBert(
        batch_size,
        args.train_data_part_num,
        args.train_data_dir,
        args.train_data_prefix,
        shuffle=False
    )
    return logits, label_ids


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalValJob():
    #8551 or 1042
    _, logits, label_ids = BuildBert(
        eval_batch_size,
        args.eval_data_part_num,
        args.eval_data_dir,
        args.eval_data_prefix,
        shuffle=False
    )
    return logits, label_ids


def run_eval_job(eval_job_func, num_steps, desc='train'):
    labels = []
    predictions = []
    for index in range(num_steps):
        logits, label = eval_job_func().get()
        predictions.extend(list(logits.numpy().argmax(axis=1)))
        labels.extend(list(label))

    def metric_fn(predictions, labels):
        return {
            "accuarcy": accuracy_score(labels, predictions), 
            "matthews_corrcoef": matthews_corrcoef(labels, predictions), 
            "precision": precision_score(labels, predictions), 
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    metric_dict = metric_fn(predictions, labels)
    print(desc, ', '.join('{}: {:.3f}'.format(k, v) for k, v in metric_dict.items()))
    #pd.DataFrame({'predictions': predictions, 'labels': labels}).to_csv('predictions_{0}.csv'.format(step), index=False)


def main():
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    InitNodes(args)

    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)
    #if args.save_and_break:
    #    print("save model just after init and exit")
    #    snapshot.save("initial_snapshot")
    #    import sys
    #    sys.exit()

    summary = Summary(args.log_dir, args)
    for epoch in range(args.num_epochs):
        metric = Metric(desc='finetune', print_steps=args.loss_print_every_n_iter, summary=summary, 
                        batch_size=batch_size, keys=['loss'])

        for step in range(epoch_size):
            BertGlueFinetuneJob().async_get(metric.metric_cb(step, epoch=epoch))
            #if 1: #step % args.loss_print_every_n_iter == 0: 

        run_eval_job(BertGlueEvalTrainJob, epoch_size, desc='train')
        run_eval_job(BertGlueEvalValJob, num_eval_steps, desc='eval')

    if args.save_last_snapshot:
        snapshot.save("last_snapshot")


if __name__ == "__main__":
    main()
