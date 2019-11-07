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

  logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*/*.log"))
  for l in logs_list:
    # extract info from file name
    model = l.strip().split('/')[-2]  # eg: vgg16
    file_name = l.strip().split('/')[-1].split('.')
    run_case = file_name[0]  # eg: 1n1c_real
    data_case = run_case.split('_')[-1] if len(run_case) > 5 else None  # eg: real or synthetic

    # extract info from file content
    tmp_dict = {
      'average_speed': 0000,
      'batch_size_per_device': 0000,
    }
    with open(l) as f:
      lines =  f.readlines()

      for line in lines:
        if "batch_size_per_device" in line:
          tmp_dict['batch_size_per_device'] = int(line.strip().split('=')[-1].strip())
        if "average speed" in line:
          tmp_dict['average_speed'] = float(line.strip().split(':')[-1].strip().split('(')[0])

    result_dict[model][run_case]['average_speed'] = tmp_dict['average_speed']
    result_dict[model][run_case]['batch_size_per_device'] = tmp_dict['batch_size_per_device']


  # print speedup
  model_list = ['vgg16', 'resnet50']
  for m in model_list:
    for c in ['_real' ,'_sythetic']:
      for s in ['1n4c']:
        speed_up = result_dict[m][s+c]['average_speed'] / result_dict[m]['1n1c'+c]['average_speed']
        # print("speedup for {}_{}{} : {}".format(m, s, c, speed_up))
        run_case = s + c
        result_dict[m][run_case]['speedup'] = round(speed_up, 2)

  pp.pprint(result_dict)
  # write to file as JSON format
  result_file_name = os.path.join(args.output_dir, "oneflow_benchmark_result.json")
  print("Saving result to {}".format(result_file_name))
  with open(result_file_name, 'w') as f:
    json.dump(result_dict, f)

if __name__ == "__main__":
  extract_result()
