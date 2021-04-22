import oneflow as flow
import oneflow_api
from oneflow.python.framework.function_util import global_function_or_identity

import numpy as np
import time
import argparse
import os
import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--model_path", type=str, default="./resnet50-19c8e357.pth", help="model path"
    )
    parser.add_argument(
        "--image_path", type=str, default="./data/fish.jpg", help="input image path"
    )
    return parser.parse_args()

def rmse(l, r):
    return np.sqrt(np.mean(np.square(l - r)))

def main(args):
    flow.env.init()
    flow.enable_eager_execution()

    start_t = time.time()
    res50_module = resnet50()
    # dic = res50_module.state_dict()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))
    # # set for eval mode
    # res50_module.eval()
    start_t = time.time()

    batch = 8
    image_nd = np.ones((batch, 3, 224, 224), dtype=np.float32)
    label_nd = np.array([e for e in range(batch)], dtype=np.int32)
    
    bp_iters = 100000000

    for i in range(bp_iters):
        print(i)

        # NOTE(Liang Depeng): cpu memory leak
        image = flow.Tensor(image_nd)
        n = image.numpy()[0]
        ################################
        
        # NOTE(Liang Depeng): gpu memory leak
        # label = flow.Tensor(label_nd, dtype=flow.int32, requires_grad=False)
        # logits = res50_module(image)
        
        # # uncomment following codes will solve the gpu memory leak
        # # grad = flow.Tensor(batch, 1000)
        # # grad.determine()
        # # @global_function_or_identity()
        # # def job():
        # #     logits.backward(grad)
        # # job()
        #####################

if __name__ == "__main__":
    args = _parse_args()
    main(args)








