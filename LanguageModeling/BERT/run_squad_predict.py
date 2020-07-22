from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from datetime import datetime
import numpy as np

import oneflow as flow

from squad import SQuADEval
import benchmark_util

parser = argparse.ArgumentParser(description="flags for bert")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument("--node_list", type=str, default=None)

# train
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay_rate", type=float, default=0.01, help="weight decay rate")
parser.add_argument("--batch_size_per_device", type=int, default=64)
parser.add_argument("--iter_num", type=int, default=1144000, help="total iterations to run")
parser.add_argument("--warmup_batches", type=int, default=10000)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--data_part_num", type=int, default=32, help="data part number in dataset")
parser.add_argument('--use_fp16', type=str2bool, nargs='?', const=True, help='use use fp16 or not')

# log and resore/save
parser.add_argument("--loss_print_every_n_iter", type=int, default=10, required=False,
    help="print loss every n iteration")
parser.add_argument("--model_save_every_n_iter", type=int, default=10000, required=False,
    help="save model every n iteration",)
parser.add_argument("--model_save_dir", type=str,
    default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
    required=False, help="model save directory")
parser.add_argument("--save_last_snapshot", type=bool, default=False, required=False,
    help="save model snapshot for last iteration")
parser.add_argument("--model_load_dir", type=str, default=None, help="model load directory")
parser.add_argument("--log_dir", type=str, default="./output", help="log info save directory")

# bert
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

args = parser.parse_args()


def BertDecoder(data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq):
    ofrecord = flow.data.ofrecord_reader(data_dir,
                                         batch_size=batch_size,
                                         data_part_num=data_part_num,
                                         random_shuffle = False,
                                         shuffle_after_epoch=False)
    blob_confs = {}
    def _blob_conf(name, shape, dtype=flow.int32):
        blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

    _blob_conf("input_ids", [seq_length])
    _blob_conf("input_mask", [seq_length])
    _blob_conf("segment_ids", [seq_length])
    _blob_conf("unique_ids", [1])

    return blob_confs


def BuildSQuADPredictNet(
    batch_size,
    data_part_num,
    seq_length=128,
    max_position_embeddings=512,
    num_hidden_layers=12,
    num_attention_heads=12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    vocab_size=30522,
    type_vocab_size=2,
    max_predictions_per_seq=20,
):
    hidden_size = 64 * num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    decoders = BertDecoder(args.data_dir, batch_size, data_part_num, seq_length,
                           max_predictions_per_seq)

    unique_ids = decoders["unique_ids"]
    input_ids = decoders["input_ids"]
    input_mask = decoders["input_mask"]
    token_type_ids = decoders["segment_ids"]


    return SQuADEval(
        unique_ids,
        input_ids,
        input_mask,
        token_type_ids,
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
        max_predictions_per_seq=max_predictions_per_seq,
        initializer_range=0.02,
    )

_BERT_MODEL_UPDATE_CONF = dict(
    learning_rate_decay=dict(
        polynomial_conf=dict(
            decay_batches=args.iter_num,
            end_learning_rate=0.0,
        )
    ),
    warmup_conf=dict(
        linear_conf=dict(warmup_batches=args.warmup_batches, start_multiplier=0,)
    ),
    clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0,)),
    adam_conf=dict(epsilon=1e-6),
    weight_decay_conf=dict(
        weight_decay_rate=args.weight_decay_rate,
        excludes=dict(pattern=["bias", "LayerNorm", "layer_norm"]),
    ),
)

config = flow.function_config()
config.default_data_type(flow.float)
config.default_distribute_strategy(flow.scope.consistent_view())

if args.use_fp16:
    config.enable_auto_mixed_precision(True)


@flow.global_function(config)
def SquadPredictJob():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    unique_ids, start_posistion, end_position = BuildSQuADPredictNet(
        batch_size,
        args.data_part_num,
        seq_length=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        vocab_size=args.vocab_size,
        type_vocab_size=args.type_vocab_size,
        max_predictions_per_seq=args.max_predictions_per_seq,
    )

    return unique_ids, start_posistion, end_position


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

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)


    if args.node_num > 1:
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
        print("Init model on demand")

    total_batch_size = (
        args.node_num * args.gpu_num_per_node * args.batch_size_per_device
    )

    all_results = []
    i = 0
    for step in range(args.iter_num):
        unique_ids, start_postion, end_postion = SquadPredictJob().get() 
        unique_ids = unique_ids.ndarray()
        start_postion = start_postion.ndarray()
        end_postion = end_postion.ndarray()
        
        for result in zip(unique_ids, start_postion, end_postion):
            all_results.append(result)
    
        if i % 100 == 0:
            print("{}/{}, num of results:{}".format(i, args.iter_num, len(all_results)))
            print("last uid:", result[0])
        i+=1
    
    np.save("all_results.npy", all_results)
    print(len(all_results), "all_resutls saved")

if __name__ == "__main__":
    main()

