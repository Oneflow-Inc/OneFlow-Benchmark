from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json
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


class AutoVivification(dict):
  """Implementation of perl's autovivification feature."""

  def __getitem__(self, item):
    try:
      return dict.__getitem__(self, item)
    except KeyError:
      value = self[item] = type(self)()
    return value


def run_model(model, gpu_per_node=1, node_num=1, node_list=None, run_real_data=True, run_synthetic_data=True):
  log_dir = os.path.join(args.output_dir, model)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  log_path = os.path.join(log_dir, "{}n{}c.log".format(node_num, gpu_per_node * node_num))
  os.system(
    "sh {}.sh {} {} {} {} {} {}".format(model, gpu_per_node, node_num, node_list, run_real_data, run_synthetic_data,
                                        log_path))
  print("Saving log to: {}".format(log_path))


def extract_result():
  result_dict = AutoVivification()

  logs_list = glob.glob(os.path.join(args.output_dir, "*/*.log*"))
  for l in logs_list:
    # extract info from file name
    model = l.strip().split('/')[-2]  # eg: vgg16
    file_name = l.strip().split('/')[-1].split('.')
    run_case = file_name[0]  # eg: 1n1c
    data_case = file_name[-1] if len(file_name) == 3 else None  # eg: real or synthetic

    # extract info from file content
    tmp_dict = {}
    with open(l) as f:
      lines =  f.readlines()
      tmp_dict['time_stamp'] = lines[0].strip().split(' ')[0] + '-' + lines[0].strip().split(' ')[1]

      for line in lines:
        if "batch_size_per_device" in line:
          tmp_dict['batch_size'] = int(line.strip().split('=')[-1].strip())
        if "average speed" in line:
          tmp_dict['average_speed'] = float(line.strip().split(':')[-1].strip().split('(')[0])


    if data_case:
      result_dict[model][run_case][data_case]['time_stamp'] = tmp_dict['time_stamp']
      result_dict[model][run_case][data_case]['average_speed'] = tmp_dict['average_speed']
      result_dict[model][run_case][data_case]['batch_size'] = tmp_dict['batch_size']
    else:
      result_dict[model][run_case]['time_stamp'] = tmp_dict['time_stamp']
      result_dict[model][run_case]['average_speed'] = tmp_dict['average_speed']
      result_dict[model][run_case]['batch_size'] = tmp_dict['batch_size']

  # write to file as JSON format
  result_file_name = os.path.join(args.output_dir, "oneflow_benchmark_result.json")
  with open(result_file_name, 'w') as f:
    json.dump(result_dict, f)


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


def main():
  benchmark()
  extract_result()


if __name__ == "__main__":
  main()
