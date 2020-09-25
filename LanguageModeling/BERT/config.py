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

import argparse
from datetime import datetime


def str_list(x):
    return x.split(',')

def int_list(x):
    return list(map(int, x.split(',')))

def float_list(x):
    return list(map(float, x.split(',')))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser(parser=None):

    parser = argparse.ArgumentParser(description="flags for bert")

    parser.add_argument('--do_train', type=str2bool, nargs='?', const=True, help='train or not')
    parser.add_argument('--do_eval', type=str2bool, nargs='?', const=True, help='eval or not')
    # resouce
    parser.add_argument("--model", type=str, default='BERT Pretrain')
    parser.add_argument("--gpu_num_per_node", type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')
    parser.add_argument("--ctrl_port", type=int, default=50051, help='ctrl_port for multinode job')

    # train
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay_rate", type=float, default=0.01, help="weight decay rate")
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument('--use_fp16', type=str2bool, nargs='?', default='False', const=True,
                        help='use use fp16 or not')
    parser.add_argument('--use_xla', type=str2bool, nargs='?', const=True,
                        help='Whether to use use xla')

    # log and resore/save
    parser.add_argument("--loss_print_every_n_iter", type=int, default=10, required=False,
        help="print loss every n iteration")
    parser.add_argument("--model_save_every_n_iter", type=int, default=10000, required=False,
        help="save model every n iteration",)
    parser.add_argument("--model_save_dir", type=str,
        default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
        required=False, help="model save directory")
    parser.add_argument("--save_last_snapshot", type=str2bool, default=False, required=False,
        help="save model snapshot for last iteration")
    parser.add_argument("--model_load_dir", type=str, default=None, help="model load directory")
    parser.add_argument("--log_dir", type=str, default="./output", help="log info save directory")

    # bert backbone
    parser.add_argument('--do_lower_case', type=str2bool, nargs='?', const=True, default='True')
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--max_predictions_per_seq", type=int, default=80)
    parser.add_argument("--num_hidden_layers", type=int, default=24)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_size_per_head", type=int, default=64)

    return parser


def print_args(args):
    print("=".ljust(66, "="))
    print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
        args.model, args.gpu_num_per_node, args.num_nodes))
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(
        str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)
