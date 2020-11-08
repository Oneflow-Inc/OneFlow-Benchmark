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
parser.add_argument("tf_model_dir", type = str, help = "Path the TensorFlow checkpoint path.")
parser.add_argument("of_dump_path", type = str, help = "Path to the output OneFlow model.")

args = parser.parse_args()

def _SaveWeightBlob2File(blob, op_name, var='out'):
    folder = os.path.join(args.of_dump_path, op_name)
    print(blob.shape, blob.dtype , folder, var)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, var)
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    f.close()
    np.save(filename, blob)

def convert():
    path = args.tf_model_dir
    init_vars = tf.train.list_variables(path)
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)
        #print("{};{};{};".format(name, shape, array.dtype))
        #print("Numpy array shape {}".format(array.shape))
        sep = name.rfind('/')
        blob_name = name[sep + 1:]
        op_name = name[:sep]
        op_name = op_name.replace('/', '-')
        if 'ln_' in op_name:
            if blob_name.endswith('b'):
                op_name = op_name + '-beta'  
            elif blob_name.endswith('g'):
                op_name = op_name + '-gamma'  
            else:
                assert 0
        elif 'c_attn' in op_name:                
            # split to q k v
            for k, v in zip(['q_attn', 'k_attn', 'v_attn'], np.split(array, 3, axis=-1)):
                new_op_name = op_name.replace('c_attn', k) + '-' + blob_name
                _SaveWeightBlob2File(v, new_op_name)
            continue
        else:
            op_name = op_name + '-' + blob_name

        _SaveWeightBlob2File(array, op_name)


if __name__ == "__main__":
    convert()

'''
of_name		tf_name	shape	dtype
model-h9-attn-c_proj-b		model/h9/attn/c_proj/b	[768]	float32
model-h9-attn-c_proj-w		model/h9/attn/c_proj/w	[1, 768, 768]	float32
model-h9-attn-k_attn-b		model/h9/attn/c_attn/b	[2304]	float32
model-h9-attn-k_attn-w		model/h9/attn/c_attn/w	[1, 768, 2304]	float32
model-h9-attn-q_attn-b				
model-h9-attn-q_attn-w				
model-h9-attn-v_attn-b				
model-h9-attn-v_attn-w				
model-h9-ln_1-beta		model/h9/ln_1/b	[768]	float32
model-h9-ln_1-gamma		model/h9/ln_1/g	[768]	float32
model-h9-ln_2-beta		model/h9/ln_2/b	[768]	float32
model-h9-ln_2-gamma		model/h9/ln_2/g	[768]	float32
model-h9-mlp-c_fc-b		model/h9/mlp/c_fc/b	[3072]	float32
model-h9-mlp-c_fc-w		model/h9/mlp/c_fc/w	[1, 768, 3072]	float32
model-h9-mlp-c_proj-b		model/h9/mlp/c_proj/b	[768]	float32
model-h9-mlp-c_proj-w		model/h9/mlp/c_proj/w	[1, 3072, 768]	float32
model-ln_f-beta		model/ln_f/b	[768]	float32
model-ln_f-gamma		model/ln_f/g	[768]	float32
model-wpe		model/wpe	[1024, 768]	float32
model-wte		model/wte	[50257, 768]	float32
'''
