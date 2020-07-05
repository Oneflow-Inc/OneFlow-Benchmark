from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image

import config as configs
parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

import oneflow as flow
from resnet_model import resnet50
from imagenet1000_clsidx_to_labels import clsidx_2_labels


def load_image(image_path='image_demo/ILSVRC2012_val_00020287.JPEG'):
    print(image_path)
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - args.rgb_mean) / args.rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')


@flow.global_function(flow.function_config())
def InferenceNet(images=flow.FixedTensorDef((1, 3, 224, 224), dtype=flow.float)):
    logits = resnet50(images, training=False, channel_last=args.channel_last)
    predictions = flow.nn.softmax(logits)
    return predictions


def main():
    flow.env.log_dir(args.log_dir)
    assert os.path.isdir(args.model_load_dir)
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    if args.channel_last:
        print("Use 'NHWC' mode >> Channel last")
    else:
        print("Use 'NCHW' mode >> Channel first")

    image = load_image()
    predictions = InferenceNet(image).get()
    clsidx = predictions.ndarray().argmax()
    print(predictions.ndarray().max(), clsidx_2_labels[clsidx])

    image = load_image(args.image_path)
    predictions = InferenceNet(image).get()
    clsidx = predictions.ndarray().argmax()
    print(predictions.ndarray().max(), clsidx_2_labels[clsidx])


if __name__ == "__main__":
    main()
