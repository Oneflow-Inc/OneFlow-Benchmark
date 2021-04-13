import oneflow as flow
import oneflow.nn as nn

import argparse
import numpy as np
from PIL import Image
import time
import torch

from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--model_path", type=str, default="./resnet50-19c8e357.pth", help="model path"
    )
    parser.add_argument(
        "--image_path", type=str, default="", help="input image path"
    )
    return parser.parse_args()

def load_image(image_path='data/tiger.jpg'):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def main(args):
    flow.env.init()
    flow.enable_eager_execution()

    start_t = time.time()
    res50_module = resnet50()
    dic = res50_module.state_dict()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    start_t = time.time()
    torch_params = torch.load(args.model_path)
    torch_keys = torch_params.keys()

    for k in dic.keys():
        if k in torch_keys:
            dic[k] = torch_params[k].detach().numpy()
    res50_module.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image)
    predictions = res50_module(image).softmax()
    predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(predictions)
    print("predict prob: %f, class name: %s" % (np.max(predictions), clsidx_2_labels[clsidx]))

if __name__ == "__main__":
    args = _parse_args()
    main(args)
