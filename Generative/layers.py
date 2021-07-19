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
import oneflow.compatible.single_client as flow

def get_const_initializer():
    return flow.constant_initializer(0.002)

def deconv2d(
    input,
    filters,
    size,
    name,
    strides=2,
    trainable=True,
    reuse=False,
    const_init=False,
    use_bias=False,
):
    name_ = name if reuse == False else name + "_reuse"
    # weight : [in_channels, out_channels, height, width]
    weight_shape = (input.shape[1], filters, size, size)
    output_shape = (
        input.shape[0],
        input.shape[1],
        input.shape[2] * strides,
        input.shape[3] * strides,
    )

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02)
        if not const_init
        else get_const_initializer(),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose(
        input,
        weight,
        strides=[strides, strides],
        output_shape=output_shape,
        padding="SAME",
        data_format="NCHW",
        name=name_,
    )

    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def conv2d(
    input,
    filters,
    size,
    name,
    strides=2,
    padding="same",
    trainable=True,
    reuse=False,
    const_init=False,
    use_bias=True,
):
    name_ = name if reuse == False else name + "_reuse"

    # (output_dim, k_h, k_w, input.shape[3]) if NHWC
    weight_shape = (filters, input.shape[1], size, size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02)
        if not const_init
        else get_const_initializer(),
        trainable=trainable,
        reuse=reuse,
    )

    output = flow.nn.compat_conv2d(
        input,
        weight,
        strides=[strides, strides],
        padding=padding,
        data_format="NCHW",
        name=name_,
    )

    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
            reuse=reuse,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def batchnorm(input, name, axis=1, reuse=False):
    name_ = name if reuse == False else name + "_reuse"
    return flow.layers.batch_normalization(input, axis=axis, name=name_)

def dense(
    input, units, name, use_bias=False, trainable=True, reuse=False, const_init=False
):
    name_ = name if reuse == False else name + "_reuse"

    in_shape = input.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    inputs = flow.reshape(input, (-1, in_shape[-1])) if in_num_axes > 2 else input

    weight = flow.get_variable(
        name="{}-weight".format(name),
        shape=(units, inputs.shape[1]),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02)
        if not const_init
        else get_const_initializer(),
        trainable=trainable,
        reuse=reuse,
        model_name="weight",
    )

    out = flow.matmul(a=inputs, b=weight, transpose_b=True, name=name_ + "matmul",)

    if use_bias:
        bias = flow.get_variable(
            name="{}-bias".format(name),
            shape=(units,),
            dtype=inputs.dtype,
            initializer=flow.random_normal_initializer()
            if not const_init
            else get_const_initializer(),
            trainable=trainable,
            reuse=reuse,
            model_name="bias",
        )
        out = flow.nn.bias_add(out, bias, name=name_ + "_bias_add")

    out = flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
    return out
