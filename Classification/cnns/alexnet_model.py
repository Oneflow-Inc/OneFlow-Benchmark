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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow

def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def _get_bias_initializer():
    return flow.zeros_initializer()

def conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=True,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
):
    if isinstance(kernel_size, int):
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    if isinstance(kernel_size, list):
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]

    weight_shape = (filters, input.shape[1], kernel_size_1, kernel_size_2)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output


def alexnet(images, need_transpose=False, channel_last=False, training=True):
    if need_transpose:
        images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    if channel_last:
    # if channel_last=True, then change mode from 'nchw' to 'nhwc'
        images = flow.transpose(images, name="transpose", perm=[0, 2, 3, 1])
    conv1 = conv2d_layer(
        "conv1", images, filters=64, kernel_size=11, strides=4, padding="VALID"
    )

    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = conv2d_layer("conv3", pool2, filters=384)

    conv4 = conv2d_layer("conv4", conv3, filters=384)

    conv5 = conv2d_layer("conv5", conv4, filters=256)

    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.nn.relu,
        use_bias=True,
        #kernel_initializer=flow.random_uniform_initializer(),
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        kernel_regularizer=_get_regularizer(),
        bias_regularizer=_get_regularizer(),
        name="fc1",
    )

    dropout1 = flow.nn.dropout(fc1, rate=0.5)

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.nn.relu,
        use_bias=True,        
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        kernel_regularizer=_get_regularizer(),
        bias_regularizer=_get_regularizer(),
        name="fc2",
    )

    dropout2 = flow.nn.dropout(fc2, rate=0.5)

    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1000,
        activation=None,
        use_bias=False,
        kernel_initializer=_get_kernel_initializer(),
        kernel_regularizer=_get_regularizer(),
        bias_initializer=False,
        name="fc3",
    )

    return fc3
