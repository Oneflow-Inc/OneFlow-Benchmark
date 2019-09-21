from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import json
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=1)
os.chdir(sys.path[0])

parser = argparse.ArgumentParser(description="flags for cnn benchmark")
parser.add_argument("--benchmark_log_dir", type=str, default="../output/benchmark_log", required=False)
parser.add_argument("--output_dir", type=str, default="./", required=False)
args = parser.parse_args()

class AutoVivification(dict):
  """Implementation of perl's autovivification feature."""

  def __getitem__(self, item):
    try:
      return dict.__getitem__(self, item)
    except KeyError:
      value = self[item] = type(self)()
    return value


def extract_result():
  result_dict = AutoVivification()

  logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*/*.log*"))
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

  pp.pprint(result_dict)

if __name__ == "__main__":
  extract_result()
