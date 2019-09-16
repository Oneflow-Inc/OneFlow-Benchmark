from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime

import oneflow as flow

import resnet_model
# import vgg_model

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("--multinode", default=False, action="store_true", required=False)
parser.add_argument("--node_list", type=str, default=" ", required=False)
parser.add_argument("--model", type=str, default="vgg16", required=False)
parser.add_argument("--batch_size", type=int, default=8, required=False)
parser.add_argument("--learning_rate", type=float, default=1e-4, required=False)
parser.add_argument("--optimizer", type=str, default="sgd", required=False)
parser.add_argument("--weight_decay", type=float, default=1e-4, required=False)
parser.add_argument("--iter_num", type=int, default=10, required=False)
parser.add_argument("--model_save_dir", type=str,
                    default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))),
                    required=False)
parser.add_argument("--model_load_dir", type=str, default=None, required=False)
parser.add_argument("--train_dir", type=str, default=None, required=False)
parser.add_argument("--data_part_num", type=int, default=32, required=False)
parser.add_argument("--eval_dir", type=str, default=None, required=False)
parser.add_argument("--skip_scp_binary", default=False, action="store_true", required=False)
parser.add_argument("--scp_binary_without_uuid", default=False, action="store_true", required=False)
parser.add_argument("--remote_by_hand", default=False, action="store_true", required=False)

args = parser.parse_args()


model_dict = {
  "resnet50": resnet_model.resnet50,
  # "vgg16":    vgg_model.vgg16,
}


@flow.function
def TrainNet():
    flow.config.train.primary_lr(args.learning_rate)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    loss = model_dict[args.model](args)
    flow.losses.add_loss(loss)
    return loss


def main():
  flow.config.default_data_type(flow.float)
  flow.config.gpu_device_num(args.gpu_num_per_node)
  flow.config.grpc_use_no_signal()
  flow.config.log_dir("./output/log")
  flow.config.ctrl_port(12138)

  if args.multinode:
    flow.config.ctrl_port(12139)
    nodes = []
    for n in args.node_list.strip().split(","):
      addr_dict = {}
      addr_dict["addr"] = n
      nodes.append(addr_dict)

    flow.config.machine(nodes)

    if args.scp_binary_without_uuid:
      flow.deprecated.init_worker(scp_binary=True, use_uuid=False)
    elif args.skip_scp_binary:
      flow.deprecated.init_worker(scp_binary=False, use_uuid=False)
    else:
      flow.deprecated.init_worker(scp_binary=True, use_uuid=True)

  check_point = flow.train.CheckPoint()
  if not args.model_load_dir:
      check_point.init()
  else:
      check_point.load(args.model_load_dir)

  num_nodes = len(args.node_list.strip().split(",")) if args.multinode else 1
  print("Traning {}: num_gpu_per_node = {}, num_nodes = {}.".format(args.model, args.gpu_num_per_node, num_nodes))

  fmt_str = "{:>12}  {:>12}  {:.6f}"
  for i in range(args.iter_num):
    train_loss = TrainNet().get().mean()
    print(fmt_str.format(i, "train loss:", train_loss))


if __name__ == '__main__':
  main()
