"""Convert tensorflow checkpoint."""

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
                    default = '.',
                    type = str,
                    required = True,
                    help = "Path the TensorFlow checkpoint path.")
parser.add_argument("--of_dump_path",
                    default = None,
                    type = str,
                    required = False,
                    help = "Path to the output OneFlow model.")

args = parser.parse_args()

def _SaveWeightBlob2File(blob, folder, var):
    #print(blob.shape, blob.dtype , folder, var)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, var)
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    #f.write(blob.tostring())
    f.close()
    #np.save(filename, blob)

def convert():
    blob_names = ['bias', 'weight', 'beta', 'gamma']
    path = args.tf_checkpoint_path
    init_vars = tf.train.list_variables(path)
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)
        #print("{};{};{};".format(name, shape, array.dtype))
        #print("Numpy array shape {}".format(array.shape))
        variable_name = name.replace('/', '-').replace('adam_', '').replace('kernel', 'weight')
        if 'classification' in name or 'cls' in name:
            continue
        print(name,variable_name,array.shape)
        if args.of_dump_path:
            #if blob_name == 'weight':
            #    array = np.transpose(array)
            folder = os.path.join(args.of_dump_path, variable_name)
            _SaveWeightBlob2File(array, folder, 'out')


if __name__ == "__main__":
    convert()

