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


from optimizer_util import add_optimizer_args
from ofrecord_util import add_ofrecord_args


def get_parser(parser=None):
    def str_list(x):
        return [i.strip() for i in x.split(',')]

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

    if parser is None:
        parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--dtype", type=str,
                        default='float32', help="float16 float32")

    # resouce
    parser.add_argument("--gpu_num_per_node", type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')
    parser.add_argument("--ctrl_port", type=int, default=50051, help='ctrl_port for multinode job')

    parser.add_argument("--model", type=str, default="resnet50",
                        help="resnet50")
    parser.add_argument(
        '--use_fp16',
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to use use fp16'
    )
    parser.add_argument(
        '--use_xla',
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to use use xla'
    )
    parser.add_argument(
        '--channel_last',
        type=str2bool,
        nargs='?',
        const=False,
        help='Whether to use use channel last mode(nhwc)'
    )
    parser.add_argument(
        '--pad_output',
        type=str2bool,
        nargs='?',
        const=True,
        help='Whether to pad the output to number of image channels to 4.'
    )

    # train and validaion
    parser.add_argument('--num_epochs', type=int,
                        default=90, help='number of epochs')
    parser.add_argument("--model_load_dir", type=str,
                        default=None, help="model load directory if need")
    parser.add_argument("--batch_size_per_device", type=int, default=64)
    parser.add_argument("--val_batch_size_per_device", type=int, default=8)

    parser.add_argument("--nccl_fusion_threshold_mb", type=int, default=0,
                        help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.")
    parser.add_argument("--nccl_fusion_max_ops", type=int, default=0,
                        help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.")

    # fuse bn relu or bn add relu
    parser.add_argument(
        '--fuse_bn_relu',
        type=str2bool,
        default=False,
        help='Whether to use use fuse batch normalization relu. Currently supported in origin/master of OneFlow only.'
    )
    parser.add_argument(
        '--fuse_bn_add_relu',
        type=str2bool,
        default=False,
        help='Whether to use use fuse batch normalization add relu. Currently supported in origin/master of OneFlow only.'
    )
    parser.add_argument("--gpu_image_decoder", type=str2bool,
                        default=False, help='Whether to use use ImageDecoderRandomCropResize.')
    # inference
    parser.add_argument("--image_path", type=str, default='test_img/tiger.jpg', help="image path")

    # for data process
    parser.add_argument("--num_classes", type=int, default=1000, help="num of pic classes")
    parser.add_argument("--num_examples", type=int,
                        default=1281167, help="train pic number")
    parser.add_argument("--num_val_examples", type=int,
                        default=50000, help="validation pic number")
    parser.add_argument('--rgb-mean', type=float_list, default=[123.68, 116.779, 103.939],
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--rgb-std', type=float_list, default=[58.393, 57.12, 57.375],
                        help='a tuple of size 3 for the std rgb')
    parser.add_argument('--image-shape', type=int_list, default=[3, 224, 224],
                        help='the image shape feed into the network')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')

    # snapshot
    parser.add_argument("--model_save_dir", type=str,
                        default="./output/snapshots/model_save-{}".format(
                            str(datetime.now().strftime("%Y%m%d%H%M%S"))),
                        help="model save directory",
                        )

    # log and loss print
    parser.add_argument("--log_dir", type=str,
                        default="./output", help="log info save directory")
    parser.add_argument(
        "--loss_print_every_n_iter",
        type=int,
        default=1,
        help="print loss every n iteration",
    )
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
    print("Time stamp: {}".format(
        str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)
