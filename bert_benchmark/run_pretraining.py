from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
from datetime import datetime

import oneflow as flow

from pretrain import PreTrain

parser = argparse.ArgumentParser(description="flags for bert")

# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument("--node_list", type=str, default=None)

# train
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_l2", type=float, default=0.01, help="weight l2 decay parameter")
parser.add_argument("--batch_size_per_device", type=int, default=24)
parser.add_argument("--iter_num", type=int, default=10, help="total iterations to run")
parser.add_argument("--warmup_iter_num", type=int, default=10, help="total iterations to run")
parser.add_argument("--log_every_n_iter", type=int, default=1, help="print loss every n iteration")
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--data_part_num", type=int, default=32, help="data part number in dataset")

# log and resore/save
parser.add_argument("--loss_print_every_n_iter", type=int, default=1, required=False,
                    help="print loss every n iteration")
parser.add_argument("--model_save_every_n_iter", type=int, default=200, required=False,
                    help="save model every n iteration")
parser.add_argument("--model_save_dir", type=str,
                    default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
                    required=False, help="model save directory")
parser.add_argument("--model_load_dir", type=str, default=None, required=False, help="model load directory")
parser.add_argument("--log_dir", type=str, default="./output", required=False, help="log info save directory")

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


def _blob_conf(name, shape, dtype=flow.int32):
  return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec())


def BertDecoder(data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq):
  blob_confs = []
  blob_confs.append(_blob_conf('input_ids', [seq_length]))
  blob_confs.append(_blob_conf('next_sentence_labels', [1]))
  blob_confs.append(_blob_conf('input_mask', [seq_length]))
  blob_confs.append(_blob_conf('segment_ids', [seq_length]))
  blob_confs.append(_blob_conf('masked_lm_ids', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_positions', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_weights', [max_predictions_per_seq], flow.float))
  return flow.data.decode_ofrecord(data_dir, blob_confs,
                                   batch_size=batch_size,
                                   name="decode",
                                   data_part_num=data_part_num)


def BuildPreTrainNet(batch_size, data_part_num, seq_length=128, max_position_embeddings=512,
                     num_hidden_layers=12, num_attention_heads=12,
                     hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                     vocab_size=30522, type_vocab_size=2, max_predictions_per_seq=20):
  hidden_size = 64 * num_attention_heads  # , H = 64, size per head
  intermediate_size = hidden_size * 4

  decoders = BertDecoder(args.data_dir, batch_size, data_part_num, seq_length,
                         max_predictions_per_seq)

  input_ids = decoders[0]
  next_sentence_labels = decoders[1]
  token_type_ids = decoders[2]
  input_mask = decoders[3]
  masked_lm_ids = decoders[4]
  masked_lm_positions = decoders[5]
  masked_lm_weights = decoders[6]
  return PreTrain(input_ids,
                  input_mask,
                  token_type_ids,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  next_sentence_labels,
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
                  initializer_range=0.02)


_BERT_MODEL_UPDATE_CONF = dict(
  learning_rate_decay=dict(
    polynomial_conf=dict(
      decay_batches=100000,
      end_learning_rate=0.0,
    )
  ),
  warmup_conf=dict(
    linear_conf=dict(
      warmup_batches=1000,
      start_multiplier=0,
    )
  ),
  clip_conf=dict(
    clip_by_global_norm=dict(
      clip_norm=1.0,
    )
  ),
  adam_conf=dict(
    epsilon=1e-6
  ),
)


@flow.function
def PretrainJob():
  total_device_num = args.node_num * args.gpu_num_per_node
  batch_size = total_device_num * args.batch_size_per_device

  flow.config.train.primary_lr(args.learning_rate)
  flow.config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
  flow.config.train.weight_l2(args.weight_l2)

  loss = BuildPreTrainNet(batch_size, args.data_part_num,
                          seq_length=args.seq_length,
                          max_position_embeddings=args.max_position_embeddings,
                          num_hidden_layers=args.num_hidden_layers,
                          num_attention_heads=args.num_attention_heads,
                          hidden_dropout_prob=args.hidden_dropout_prob,
                          attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                          vocab_size=args.vocab_size,
                          type_vocab_size=args.type_vocab_size,
                          max_predictions_per_seq=args.max_predictions_per_seq)
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  print("=".ljust(66, '='))
  print("Running bert: num_gpu_per_node = {}, num_nodes = {}.".format(
                                 args.gpu_num_per_node, args.node_num))
  print("=".ljust(66, '='))
  for arg in vars(args):
    print('{} = {}'.format(arg, getattr(args, arg)))
  print("-".ljust(66, '-'))
  print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

  flow.config.gpu_device_num(args.gpu_num_per_node)
  flow.config.ctrl_port(19009)
  flow.config.default_data_type(flow.float)
  flow.config.log_dir(args.log_dir)

  if args.node_num > 1:
    flow.config.ctrl_port(19008)
    nodes = []
    for n in args.node_list.strip().split(","):
      addr_dict = {}
      addr_dict["addr"] = n
      nodes.append(addr_dict)

    flow.config.machine(nodes)

  check_point = flow.train.CheckPoint()
  if args.model_load_dir:
    assert os.path.isdir(args.model_load_dir)
    check_point.load(args.model_load_dir)
    print('Restoring model from {}.'.format(args.model_load_dir))
  else:
    check_point.init()
    print('Init model on demand')

  # warmups
  print("Runing warm up for {} iterations.".format(args.warmup_iter_num))
  for step in range(args.warmup_iter_num):
    train_loss = PretrainJob().get().mean()

  print("Start trainning.")
  total_time = 0.0
  batch_size = args.node_num * args.gpu_num_per_node * args.batch_size_per_device
  for step in range(args.iter_num):
    start_time = time.time()
    train_loss = PretrainJob().get().mean()
    duration = time.time() - start_time
    total_time += duration

    if step % args.loss_print_every_n_iter == 0:
      images_per_sec = batch_size / duration
      print("iter {}, loss: {:.3f}, speed: {:.3f}(sec/batch), {:.3f}(sentencs/sec)"
            .format(step, train_loss, duration, images_per_sec))

    if (step + 1) % args.model_save_every_n_iter == 0:
      if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
        snapshot_save_path = os.path.join(args.model_save_dir, 'snapshot_%d' % (step + 1))
        print("Saving model to {}.".format(snapshot_save_path))
        check_point.save(snapshot_save_path)

  avg_img_per_sec = batch_size * args.iter_num / total_time
  print("-".ljust(66, '-'))
  print("average speed: {:.3f}(sentences/sec)".format(avg_img_per_sec))
  print("-".ljust(66, '-'))

