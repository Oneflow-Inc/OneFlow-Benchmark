from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import logging

from dali_util import add_dali_args
from optimizer_util import add_optimizer_args
from ofrecord_util import add_ofrecord_args

def get_parser(parser=None):
    def str_list(x):
        return x.split(',')
    def int_list(x):
        return list(map(int, x.split(',')))
    def float_list(x):
        return list(map(float, x.split(',')))

    if parser is None:
        parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--dtype", type=str, default='float32', help="float16 float32")

    # resouce
    parser.add_argument("--gpu_num_per_node", type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')

    parser.add_argument("--model", type=str, default="vgg16", help="vgg16 or resnet50")
    parser.add_argument("--use_fp16", type=bool, default=False, help="fp16")

    # train and validaion
    parser.add_argument('--num_epochs', type=int, default=90, help='number of epochs')
    parser.add_argument("--model_load_dir", type=str, default=None, help="model load directory if need")
    parser.add_argument("--batch_size_per_device", type=int, default=64)
    parser.add_argument("--val_batch_size_per_device", type=int, default=8)

    # for data process 
    parser.add_argument("--num_examples", type=int, default=1281167, help="train pic number")
    parser.add_argument("--num_val_examples", type=int, default=50000, help="validation pic number")
    parser.add_argument('--rgb-mean', type=float_list, default=[123.68, 116.779, 103.939],
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--rgb-std', type=float_list, default=[58.393, 57.12, 57.375],
                        help='a tuple of size 3 for the std rgb')
    parser.add_argument("--input_layout", type=str, default='NHWC', help="NCHW or NHWC")


    ## snapshot
    parser.add_argument("--model_save_dir", type=str,
        default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
        help="model save directory",
    )

    # log and loss print
    parser.add_argument("--log_dir", type=str, default="./output", help="log info save directory")
    parser.add_argument(
        "--loss_print_every_n_iter",
        type=int,
        default=1,
        help="print loss every n iteration",
    )
    add_dali_args(parser)
    add_ofrecord_args(parser)
    add_optimizer_args(parser)
    return parser


def print_args(args):
    print("=".ljust(66, "="))
    print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.model, args.gpu_num_per_node, args.num_nodes))
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)
