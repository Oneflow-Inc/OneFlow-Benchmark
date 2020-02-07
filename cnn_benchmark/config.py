import argparse
from datetime import datetime

from dali import add_dali_args

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
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.15', '192.168.1.16'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')

    parser.add_argument("--model", type=str, default="vgg16", help="vgg16 or resnet50")

    # train
    parser.add_argument("--model_load_dir", type=str, default=None, help="model load directory if need")
    parser.add_argument("--batch_size_per_device", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="momentum-cosine-decay",
                        help="sgd, adam, momentum, momentum-cosine-decay")
    parser.add_argument("--weight_l2", type=float, default=None, help="weight decay parameter")
    parser.add_argument("--train_step_num", type=int, default=10, help="total training step number")
    parser.add_argument("--data_dir", type=str, default=None, help="training dataset directory")
    parser.add_argument("--data_part_num", type=int, default=32, help="training data part number")
    parser.add_argument("--image_size", type=int, default=224, help="image size")#Todo, remove

    # from mxnet
    parser.add_argument('--num_epochs', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr-schedule', choices=('multistep', 'cosine'), default='cosine',
                        help='learning rate schedule')
    parser.add_argument('--lr-factor', type=float, default=0.256,
                        help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-steps', type=float_list, default=[],
                        help='the epochs to reduce the lr, e.g. 30,60')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='the epochs to ramp-up lr to scaled large-batch value')

    parser.add_argument("--input_layout", type=str, default='NHWC', help="NCHW or NHWC")
    parser.add_argument('--image-shape', type=int_list, default=[3, 224, 224],
                        help='the image shape feed into the network')
    parser.add_argument('--rgb-mean', type=float_list, default=[123.68, 116.779, 103.939],
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--rgb-std', type=float_list, default=[58.393, 57.12, 57.375],
                        help='a tuple of size 3 for the std rgb')
    parser.add_argument('--data_train', type=str, help='the training data')
    parser.add_argument('--data_train_idx', type=str, default='', help='the index of training data')
    parser.add_argument('--data_val', type=str, help='the validation data')
    parser.add_argument('--data_val_idx', type=str, default='', help='the index of validation data')
    parser.add_argument("--num_examples", type=int, default=1281167, help="imagenet pic number")

    ## snapshot
    parser.add_argument(
        "--model_save_every_n_iter",
        type=int,
        default=200,
        help="save model every n iteration",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
        help="model save directory",
    )

    # validation
    parser.add_argument("--val_step_num", type=int, default=10, help="total validation step number")
    parser.add_argument("--val_batch_size_per_device", type=int, default=8)
    parser.add_argument("--val_data_dir", type=str, default=None, help="validation dataset directory")
    parser.add_argument("--val_data_part_num", type=int, default=32, help="validation data part number")

    # log and loss print
    parser.add_argument("--log_dir", type=str, default="./output", help="log info save directory")
    parser.add_argument(
        "--loss_print_every_n_iter",
        type=int,
        default=1,
        help="print loss every n iteration",
    )
    parser.add_argument(
        "--val_print_every_n_iter",
        type=int,
        default=10,
        help="print loss every n iteration",
    )
    add_dali_args(parser)
    return parser


if __name__ == '__main__':
    parser = get_parser(None)
    config = parser.parse_known_args()
    print(config)
