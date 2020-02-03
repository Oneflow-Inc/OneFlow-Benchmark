import argparse
from datetime import datetime

def get_parser(parser=None):
    def str_list(x):
        return x.split(',')
    def int_list(x):
        return list(map(int, x.split(',')))
    def float_list(x):
        return list(map(float, x.split(',')))

    if parser is None:
        parser = argparse.ArgumentParser("flags for cnn benchmark")

    # resouce
    parser.add_argument("--gpu_num_per_node", type=int, default=1)
    parser.add_argument("--node_num", type=int, default=1)
    parser.add_argument("--node_list", type=str, default=None, help="nodes' IP address, split by comma")

    parser.add_argument("--model", type=str, default="vgg16", help="vgg16 or resnet50")

    # train
    parser.add_argument("--model_load_dir", type=str, default=None, help="model load directory if need")
    parser.add_argument("--batch_size_per_device", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd, adam, momentum")
    parser.add_argument("--weight_l2", type=float, default=None, help="weight decay parameter")
    parser.add_argument("--train_step_num", type=int, default=10, help="total training step number")
    parser.add_argument("--data_dir", type=str, default=None, help="training dataset directory")
    parser.add_argument("--data_part_num", type=int, default=32, help="training data part number")
    parser.add_argument("--image_size", type=int, default=228, help="image size")

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

    return parser


if __name__ == '__main__':
    parser = get_parser(None)
    config = parser.parse_known_args()
    print(config)
