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
import argparse
from datetime import datetime

import config as configs
from config import str2bool
import oneflow as flow

from squad import SQuAD
from util import Snapshot, Summary, InitNodes, Metric, CreateOptimizer, GetFunctionConfig
from squad_util import RawResult, gen_eval_predict_json

parser = configs.get_parser()
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument("--train_data_dir", type=str, default=None)
parser.add_argument("--train_example_num", type=int, default=88614, 
                    help="example number in dataset")
parser.add_argument("--batch_size_per_device", type=int, default=32)
parser.add_argument("--train_data_part_num", type=int, default=1, 
                    help="data part number in dataset")
parser.add_argument("--eval_data_dir", type=str, default=None)
parser.add_argument("--eval_example_num", type=int, default=10833, 
                    help="example number in dataset")
parser.add_argument("--eval_batch_size_per_device", type=int, default=64)
parser.add_argument("--eval_data_part_num", type=int, default=1, 
                    help="data part number in dataset")

# post eval
parser.add_argument("--output_dir", type=str, default='squad_output', help='folder for output file')
parser.add_argument("--doc_stride", type=int, default=128)
parser.add_argument("--max_seq_length", type=int, default=384)
parser.add_argument("--max_query_length", type=int, default=64)
parser.add_argument("--vocab_file", type=str,
                    help="The vocabulary file that the BERT model was trained on.")
parser.add_argument("--predict_file", type=str, 
                    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument("--n_best_size", type=int, default=20,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", type=int, default=30,
    help="The maximum length of an answer that can be generated. This is needed \
    because the start and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", type=str2bool, default='False',
    help="If true, all of the warnings related to data processing will be printed. \
    A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument("--version_2_with_negative", type=str2bool, default='False',
    help="If true, the SQuAD examples contain some that do not have an answer.")
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.")

args = parser.parse_args()

batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device

epoch_size = math.ceil(args.train_example_num / batch_size)
num_eval_steps = math.ceil(args.eval_example_num / eval_batch_size)
args.iter_num = epoch_size * args.num_epochs
args.predict_batch_size = eval_batch_size
configs.print_args(args)

def SquadDecoder(data_dir, batch_size, data_part_num, seq_length, is_train=True):
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             random_shuffle = is_train,
                                             shuffle_after_epoch=is_train)
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


if args.do_train:
    @flow.global_function(type='train', function_config=GetFunctionConfig(args))
    def SquadFinetuneJob():
        hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
        intermediate_size = hidden_size * 4
    
        decoders = SquadDecoder(args.train_data_dir, batch_size, args.train_data_part_num, args.seq_length)
    
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
        opt = CreateOptimizer(args)
        opt.minimize(total_loss)
        return {'total_loss': total_loss}
    
if args.do_eval:
    @flow.global_function(type='predict')
    def SquadDevJob():
        hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
        intermediate_size = hidden_size * 4
    
        decoders = SquadDecoder(args.eval_data_dir, eval_batch_size, args.eval_data_part_num, args.seq_length,
                                is_train=False)
    
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
    
        return decoders['unique_ids'], start_logits, end_logits


def main():
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    InitNodes(args)

    if args.do_train or args.do_eval:
        snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    if args.do_train:
        summary = Summary(args.log_dir, args)
        for epoch in range(args.num_epochs):
            metric = Metric(desc='train', print_steps=args.loss_print_every_n_iter, summary=summary, 
                            batch_size=batch_size, keys=['total_loss'])

            for step in range(epoch_size):
                SquadFinetuneJob().async_get(metric.metric_cb(step, epoch=epoch))

        if args.save_last_snapshot:
            snapshot.save("last_snapshot")
    
    if args.do_eval:
        assert os.path.isdir(args.eval_data_dir)
        all_results = []
        for step in range(num_eval_steps):
            unique_ids, start_positions, end_positions = SquadDevJob().get()
            unique_ids = unique_ids.numpy()
            start_positions = start_positions.numpy()
            end_positions = end_positions.numpy()
        
            for unique_id, start_position, end_position in zip(unique_ids, start_positions, end_positions):
                all_results.append(RawResult(
                    unique_id = int(unique_id[0]),
                    start_logits = start_position.flatten().tolist(),
                    end_logits = end_position.flatten().tolist(),
                ))
    
            if step % args.loss_print_every_n_iter == 0:
                print("{}/{}, num of results:{}".format(step, num_eval_steps, len(all_results)))
                print("last uid:", unique_id[0])
        
        gen_eval_predict_json(args, all_results)


if __name__ == "__main__":
    main()
