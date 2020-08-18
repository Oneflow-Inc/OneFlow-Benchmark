"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import OrderedDict
import os
from PIL import Image
import time
from typing import Callable, Text

import numpy as np
import oneflow as flow
import oneflow.typing as tp
import onnx
import onnxruntime as ort

from resnet_model import resnet50
from imagenet1000_clsidx_to_labels import clsidx_2_labels


def load_image(image_path: Text) -> np.ndarray:
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    print(image_path)
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')


@flow.global_function("predict")
def InferenceNet(images: tp.Numpy.Placeholder((1, 3, 224, 224), dtype=flow.float)) -> tp.Numpy:
    logits = resnet50(images, training=False)
    predictions = flow.nn.softmax(logits)
    return predictions


def onnx_inference(image: np.ndarray, onnx_model: onnx.ModelProto):
    """
    test onnx model with onnx runtime
    :param image:           input image, a numpy array
    :param onnx_model:      onnx model
    :return:
    """
    assert os.path.isfile(image_path)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    assert len(sess.get_outputs()) == 1 and len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_dict[ipt.name] = image
    onnx_res = sess.run([], ipt_dict)[0]
    return onnx_res


def oneflow_to_onnx(job_func: Callable, flow_weights_path: Text, onnx_model_dir: Text, external_data: bool=False):
    """
    convert oneflow model to onnx model
    :param job_func:            inference function in oneflow
    :param flow_weights_path:   input oneflow model path
    :param onnx_model_dir:      output dir path to save model.onnx
    :return: onnx model
    """
    if not os.path.exists(onnx_model_dir): os.makedirs(onnx_model_dir)
    assert os.path.exists(flow_weights_path) and os.path.isdir(onnx_model_dir)

    onnx_model_path = os.path.join(onnx_model_dir, os.path.basename(flow_weights_path) + '.onnx')
    flow.onnx.export(job_func, flow_weights_path, onnx_model_path, opset=11, external_data=external_data)
    print('Convert to onnx success! >> ', onnx_model_path)
    return onnx.load_model(onnx_model_path)


def check_equality(job_func: Callable, onnx_model: onnx.ModelProto, image_path: Text) -> (bool, np.ndarray):
    image = load_image(image_path)
    onnx_res = onnx_inference(image, onnx_model)
    oneflow_res = job_func(image)
    is_equal = np.allclose(onnx_res, oneflow_res, rtol=1e-4, atol=1e-5)
    return is_equal, onnx_res


if __name__ == "__main__":
    image_path = 'data/tiger.jpg'
    # set up your model path
    flow_weights_path = 'resnet_v15_of_best_model_val_top1_77318'
    onnx_model_dir = 'onnx/model'

    check_point = flow.train.CheckPoint()
    check_point.load(flow_weights_path)

    # conver oneflow to onnx
    onnx_model = oneflow_to_onnx(InferenceNet, flow_weights_path, onnx_model_dir, external_data=False)

    # check equality
    are_equal, onnx_res = check_equality(InferenceNet, onnx_model, image_path)
    clsidx_onnx = onnx_res.argmax()
    print('Are the results equal? {}'.format('Yes' if are_equal else 'No'))
    print('Class: {}; score: {}'.format(clsidx_2_labels[clsidx_onnx], onnx_res.max()))
