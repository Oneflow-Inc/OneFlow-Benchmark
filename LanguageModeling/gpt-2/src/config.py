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
    parser = argparse.ArgumentParser(description='Train GPT-2 on your custom dataset.')
    
    parser.add_argument('--dataset', metavar='PATH', type=str, required=True, 
                        help='Input file, directory (utf-8 text, or preencoded .npz files).')
    parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, 
                        help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
    parser.add_argument('--encoding', type=str, default='utf-8', 
                        help='Set the encoding for reading and writing files.')
    parser.add_argument('--cfg_dir', metavar='CONFIG', type=str, default='models/117M', 
                        help='folder contains `hparams.json`, `encoder.json` and `vocab.bpe`')
                    
    parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
    parser.add_argument('--seq_len', metavar='SEQUENCE', type=int, default=1024, help='sequence length')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
    parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, 
                        help='Learning rate for Adam')
    
    parser.add_argument("--gpu_num_per_node", type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')
    parser.add_argument("--ctrl_port", type=int, default=50051, help='ctrl_port for multinode job')
    
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
    
    return parser


def print_args(args):
    print("=".ljust(66, "="))
    print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
        'GPT-2', args.gpu_num_per_node, args.num_nodes))
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
