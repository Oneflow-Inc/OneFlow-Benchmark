"""Convert tensorflow checkpoint to oneflow snapshot"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import numpy as np
import os

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--tf_checkpoint_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path the TensorFlow checkpoint path.")
parser.add_argument("--of_dump_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path to the output OneFlow model.")

args = parser.parse_args()

def _SaveWeightBlob2File(blob, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, "out")
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    f.close()

def convert():
    path = args.tf_checkpoint_path
    init_vars = tf.train.list_variables(path)
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)

        sep = name.rfind('/')
        blob_name = name[sep + 1:]
        op_name = name[:sep].replace('/', '-')

        if blob_name == "kernel":
            blob_name = "weight"
        elif blob_name in ['adam_m', 'adam_v']:
            print("find m, v weights")

        folder_name = op_name+"-"+blob_name
        folder = os.path.join(args.of_dump_path, folder_name)
        print("saved to:", folder)

        _SaveWeightBlob2File(array, folder)


if __name__ == "__main__":
    convert()

