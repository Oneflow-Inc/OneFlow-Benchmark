# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import oneflow as flow
import onnxruntime as ort
import numpy as np

import time
from PIL import Image
from collections import OrderedDict

from resnet_model import resnet50
from imagenet1000_clsidx_to_labels import clsidx_2_labels


def load_image(image_path):
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


def func_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    return func_config


@flow.global_function(func_config())
def InferenceNet(images=flow.FixedTensorDef((1, 3, 224, 224))):
    logits = resnet50(images, training=False)
    predictions = flow.nn.softmax(logits)
    return predictions


def onnx_inference(image_path, onnx_model_path, ort_optimize=True):
    """
    test onnx model with onnx runtime
    :param image_path:      input image path
    :param onnx_model_path: path of model.onnx
    :param ort_optimize:
    :return:
    """
    assert os.path.isfile(image_path) and os.path.isfile(onnx_model_path)
    ort_sess_opt = ort.SessionOptions()
    ort_sess_opt.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if ort_optimize else \
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_model_path, sess_options=ort_sess_opt)
    assert len(sess.get_outputs()) == 1 and len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_dict[ipt.name] = load_image(image_path)
    start = time.time()
    onnx_res = sess.run([], ipt_dict)[0]
    print('Cost: %.4f s' % (time.time() - start))
    clsidx_onnx = onnx_res.argmax()
    print('Onnx >> ', onnx_res.max(), clsidx_2_labels[clsidx_onnx])


def oneflow_to_onnx(job_func, flow_model_path, onnx_model_dir, external_data=False):
    """
    convert oneflow model to onnx model
    :param job_func:        inference function in oneflow
    :param flow_model_path: input oneflow model path
    :param onnx_model_dir:  output dir path to save model.onnx
    :param external_data:
    :return: ture or false
    """
    if not os.path.exists(onnx_model_dir): os.makedirs(onnx_model_dir)
    assert os.path.exists(flow_model_path) and os.path.isdir(onnx_model_dir)

    check_point = flow.train.CheckPoint()
    # it is a trick to keep check_point.save() from hanging when there is no variable
    @flow.global_function(flow.FunctionConfig())
    def add_var():
        return flow.get_variable(
            name="trick",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
    check_point.init()

    onnx_model_path = os.path.join(onnx_model_dir, os.path.basename(flow_model_path) + '.onnx')
    flow.onnx.export(job_func, flow_model_path, onnx_model_path, opset=11, external_data=external_data)
    print('Convert to onnx success! >> ', onnx_model_path)
    return onnx_model_path


if __name__ == "__main__":
    # path = 'tiger.jpg'
    path = 'test_img/ILSVRC2012_val_00020287.JPEG'
    flow_model_path = '/your/oneflow/model/path'
    onnx_model_dir = 'onnx/model'

    # conver oneflow to onnx
    onnx_model_path = oneflow_to_onnx(InferenceNet, flow_model_path, onnx_model_dir, external_data=False)

    # inference
    onnx_inference(path, onnx_model_path)

    # Output:
    # ILSVRC2012_val_00020287.JPEG
    # Cost: 0.0319s
    # Onnx >> 0.9924272 hay
