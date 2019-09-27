from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from datetime import datetime

import oneflow as flow

import resnet_model
import vgg_model

parser = argparse.ArgumentParser(description="flags for cnn benchmark")

# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument("--node_list", type=str, default=None, required=False, help="nodes' IP address, split by comma")

# train
parser.add_argument("--model", type=str, default="vgg16", required=False, help="vgg16 or resnet50")
parser.add_argument("--batch_size_per_device", type=int, default=8, required=False)
parser.add_argument("--learning_rate", type=float, default=1e-4, required=False)
parser.add_argument("--optimizer", type=str, default="sgd", required=False, help="sgd, adam, momentum")
parser.add_argument("--weight_l2", type=float, default=None, required=False, help="weight decay parameter")
parser.add_argument("--iter_num", type=int, default=10, required=False, help="total iterations to run")
parser.add_argument("--warmup_iter_num", type=int, default=0, required=False, help="total iterations to run")
parser.add_argument("--data_dir", type=str, default=None, required=False, help="dataset directory")
parser.add_argument("--data_part_num", type=int, default=32, required=False, help="data part number in dataset")

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

args = parser.parse_args()

model_dict = {
  "resnet50": resnet_model.resnet50,
  "vgg16": vgg_model.vgg16,
}

optimizer_dict = {
  "sgd": {"naive_conf": {}},
  "adam": {"adam_conf": {"beta1": 0.9}},
  "momentum": {"momentum_conf": {"beta": 0.9}},
}


@flow.function
def TrainNet():
  flow.config.train.primary_lr(args.learning_rate)
  flow.config.train.model_update_conf(optimizer_dict[args.optimizer])
  if args.weight_l2:
    flow.config.train.weight_l2(args.weight_l2)

  loss = model_dict[args.model](args)
  flow.losses.add_loss(loss)
  return loss


def main():
  print("=".ljust(66, '='))
  print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
                 args.model, args.gpu_num_per_node, args.node_num))
  print("=".ljust(66, '='))
  for arg in vars(args):
    print('{} = {}'.format(arg, getattr(args, arg)))
  print("-".ljust(66, '-'))
  print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))
  flow.config.default_data_type(flow.float)
  flow.config.gpu_device_num(args.gpu_num_per_node)
  flow.config.grpc_use_no_signal()
  flow.config.log_dir(args.log_dir)
  flow.config.ctrl_port(12139)

  if args.node_num > 1:
    flow.config.ctrl_port(12138)
    nodes = []
    for n in args.node_list.strip().split(","):
      addr_dict = {}
      addr_dict["addr"] = n
      nodes.append(addr_dict)

    flow.config.machine(nodes)

  check_point = flow.train.CheckPoint()
  if args.model_load_dir:
    assert os.path.isdir(args.model_load_dir)
    print("Restoring model from {}.".format(args.model_load_dir))
    check_point.load(args.model_load_dir)
  else:
    print("Init model on demand.")
    check_point.init()

  # warmups
  print("Runing warm up for {} iterations.".format(args.warmup_iter_num))
  for step in range(args.warmup_iter_num):
    train_loss = TrainNet().get().mean()

  print("Start trainning.")
  main.total_time = 0.0
  main.batch_size = args.node_num * args.gpu_num_per_node * args.batch_size_per_device
  main.start_time = time.time()
  def create_callback(step):
    def callback(train_loss):
      if step % args.loss_print_every_n_iter == 0:
        cur_time = time.time()
        duration = cur_time - main.start_time
        main.total_time += duration
        main.start_time = cur_time
        images_per_sec = main.batch_size / duration
        print("iter {}, loss: {:.3f}, speed: {:.3f}(sec/batch), {:.3f}(images/sec)"
              .format(step, train_loss.mean(), duration, images_per_sec))

    return callback

  for step in range(args.iter_num):
    TrainNet().async_get(create_callback(step))

    if (step + 1) % args.model_save_every_n_iter == 0:
      if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
        snapshot_save_path = os.path.join(args.model_save_dir, 'snapshot_%d' % (step + 1))
        print("Saving model to {}.".format(snapshot_save_path))
        check_point.save(snapshot_save_path)

  avg_img_per_sec = main.batch_size * args.iter_num / main.total_time
  print("-".ljust(66, '-'))
  print("average speed: {:.3f}(images/sec)".format(avg_img_per_sec))
  print("-".ljust(66, '-'))

if __name__ == '__main__':
  main()
