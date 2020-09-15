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

import oneflow as flow

def _batch_norm(inputs, name=None, trainable=True, data_format="NCHW"):
    axis = 1 if  data_format=="NCHW" else 3
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=0.997,
        epsilon=1.001e-5,
        center=True,
        scale=True,
        trainable=trainable,
        name=name,
    )

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def conv2d_layer(
    name,
    input,
    filters,
    weight_initializer,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=True,
    bias_initializer=flow.zeros_initializer(),

    weight_regularizer=_get_regularizer(), # weight_decay
    bias_regularizer=_get_regularizer(),

    bn=True,
): 
    weight_shape =  (filters, input.shape[1], kernel_size, kernel_size) if data_format=="NCHW"  else  (filters, kernel_size, kernel_size, input.shape[3])
    weight = flow.get_variable(
        name + "_weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            if bn:
                output = _batch_norm(output, name + "_bn", True, data_format)
                output = flow.nn.relu(output)
            else:
                output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output


def _conv_block(in_blob, index, filters, conv_times, data_format="NCHW"):
    conv_block = []
    conv_block.insert(0, in_blob)
    weight_initializer = flow.variance_scaling_initializer(2, 'fan_out', 'random_normal', data_format=data_format)
    for i in range(conv_times):
        conv_i = conv2d_layer(
            name="conv{}".format(index),
            input=conv_block[i],
            filters=filters,
            kernel_size=3,
            strides=1,
            data_format=data_format,
            weight_initializer=weight_initializer,
            bn=True,
        )

        conv_block.append(conv_i)
        index += 1

    return conv_block

def vgg16bn(images, args, trainable=True, training=True):
    data_format="NHWC" if args.channel_last else "NCHW"
    
    conv1 = _conv_block(images, 0, 64, 2, data_format)
    pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", data_format, name="pool1")
    
    conv2 = _conv_block(pool1, 2, 128, 2, data_format)
    pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", data_format, name="pool2")

    conv3 = _conv_block(pool2, 4, 256, 3, data_format)
    pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", data_format, name="pool3")

    conv4 = _conv_block(pool3, 7, 512, 3, data_format)
    pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, "VALID", data_format, name="pool4")

    conv5 = _conv_block(pool4, 10, 512, 3, data_format)
    pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", data_format, name="pool5")

    def _get_kernel_initializer():
        return flow.random_normal_initializer(stddev=0.01)

    def _get_bias_initializer():
        return flow.zeros_initializer()

    pool5 = flow.reshape(pool5, [pool5.shape[0], -1])
    fc6 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.nn.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        kernel_regularizer=_get_regularizer(),  # weght_decay
        bias_regularizer=_get_regularizer(),
        trainable=trainable,
        name="dense0",
    )

    fc6 = flow.nn.dropout(fc6, rate=0.5)

    fc7 = flow.layers.dense(
        inputs=fc6,
        units=4096,
        activation=flow.nn.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="dense1",
    )
    fc7 = flow.nn.dropout(fc7, rate=0.5)

    fc8 = flow.layers.dense(
        inputs=fc7,
        units=1000,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="dense2",
    )

    return fc8
