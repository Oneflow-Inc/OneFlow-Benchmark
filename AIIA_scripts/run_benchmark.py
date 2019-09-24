from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

parser = argparse.ArgumentParser(description="flags for cnn benchmark")
parser.add_argument("--model", type=str, default="vgg16", required=False,
                    help="model(s) to run, split by comma")
parser.add_argument("--case", type=str, default="1n1c", required=False, help="cases to run, split by comma")
parser.add_argument("--node_list", type=str, default=None, required=False, help="nodes' IP address, split by comma")
parser.add_argument("--run_real_data", type=bool, default=True, required=False)
parser.add_argument("--run_synthetic_data", type=bool, default=True, required=False)
parser.add_argument("--output_dir", type=str, default="./output/benchmark_log", required=False)
args = parser.parse_args()


def run_model(model, gpu_per_node=1, node_num=1, node_list=None, run_real_data=True, run_synthetic_data=True):
  log_dir = os.path.join(args.output_dir, model)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  log_path = os.path.join(log_dir, "{}n{}c.log".format(node_num, gpu_per_node * node_num))
  os.system(
    "sh {}.sh {} {} {} {} {} {}".format(model, gpu_per_node, node_num, node_list, run_real_data, run_synthetic_data,
                                        log_path))
  print("Saving log to: {}".format(log_path))


def benchmark():
  model_list = args.model.strip().split(',')
  assert len(model_list) > 0

  case_list = args.case.strip().split(',')
  assert len(case_list) > 0

  for model in model_list:
    model = model.strip()
    for case in case_list:
      case = case.strip()
      node_num = int(case[0])
      gpu_per_node = int(int(case[2]) / node_num)

      if node_num > 1:
        node_list = args.node_list.strip().split(',')
        assert len(node_list) == node_num

      run_model(model, gpu_per_node=gpu_per_node, node_num=node_num, node_list=args.node_list,
                run_real_data=args.run_real_data, run_synthetic_data=args.run_real_data)


if __name__ == "__main__":
  benchmark()
