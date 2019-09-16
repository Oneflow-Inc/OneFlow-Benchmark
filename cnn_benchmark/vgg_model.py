from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

import data_loader

TRAINABLE = True
IMAGE_SIZE = 224

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=True,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.constant_initializer(),
):
    weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
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
        )
        output = flow.nn.bias_add(output, bias, "NCHW")
    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _conv_block(in_blob, index, filters, conv_times):
    conv_block = []
    conv_block.insert(0, in_blob)
    for i in range(conv_times):
        conv_i = _conv2d_layer(
            name="conv{}".format(index),
            input=conv_block[i],
            filters=filters,
            kernel_size=3,
            strides=1,
        )
        conv_block.append(conv_i)
        index += 1

    return conv_block


def vgg16(args):

    (labels, images) = data_loader.load_imagenet(args.data_dir, IMAGE_SIZE, args.batch_size, args.data_part_num)

    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    conv1 = _conv_block(transposed, 0, 64, 2)
    pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv_block(pool1, 2, 128, 2)
    pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv_block(pool2, 4, 256, 3)
    pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", "NCHW", name="pool3")

    conv4 = _conv_block(pool3, 7, 512, 3)
    pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, "VALID", "NCHW", name="pool4")

    conv5 = _conv_block(pool4, 10, 512, 3)
    pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", "NCHW", name="pool5")

    def _get_kernel_initializer():
        kernel_initializer = op_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer

    def _get_bias_initializer():
        bias_initializer = op_conf_util.InitializerConf()
        bias_initializer.constant_conf.value = 0.0
        return bias_initializer

    pool5 = flow.reshape(pool5, [pool5.shape[0], -1])

    fc6 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=TRAINABLE,
        name="fc1"
    )

    fc7 = flow.layers.dense(
        inputs=fc6,
        units=4096,
        activation=flow.keras.activations.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=TRAINABLE,
        name="fc2"
    )
    fc7 = flow.nn.dropout(fc7, rate=0.5)


    fc8 = flow.layers.dense(
        inputs=fc7,
        units=1001,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=TRAINABLE,
        name="fc_final"
    )
    fc8 = flow.nn.dropout(fc8, rate=0.5)

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, fc8, name="softmax_loss"
    )

    return loss
