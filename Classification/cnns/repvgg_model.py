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
import math


# TODO: Our default conv2d initializer is Xaview_uniform
# However, pytorch default initializer is Kaiming_uniform


def _get_regularizer(model_name):
    """
    Only use regularizer in ConvLayer.
    :param model_name: The layer name.
    :return: A regularizer with l2(0.0001)
    """
    if model_name == "bias" or model_name == "bn":
        return None
    else:
        # Other params weight_decay = 1e-4
        return flow.regularizers.l2(0.0001)


def torch_style_conv(x, out_channels, kernel_size, stride, padding, groups=1, use_bias=False, trainable=True,
                     name="conv_"):
    """
    The Conv2D Layer
    :param x: The input Tensor.
    :param out_channels: The output channels.
    :param kernel_size: The kernelsize of ConvLayer.
    :param stride: The stride of ConvLayer.
    :param padding: The padding.
    :param groups: The groups of ConvLayer.
    :param use_bias: Whether use bias in ConvLayer.
    :param trainable: Whether the conv weight is trained.
    :param name: The name for the ConvLayer.
    :return: The output Tensor.
    """
    _channel = x.shape[1]  # NCHW
    weight_regularizer = _get_regularizer("weight")
    weight_shape = (out_channels, _channel // groups, kernel_size, kernel_size)
    weight_init = flow.kaiming_initializer(
        shape=weight_shape, distribution="random_uniform", negative_slope=math.sqrt(5)
    )
    _conv_weight = flow.get_variable(
        name=name + "weight",
        shape=weight_shape,
        initializer=weight_init,
        regularizer=weight_regularizer,
        trainable=trainable
    )
    _conv_out = flow.nn.conv2d(x, _conv_weight, stride, padding, groups=groups, name=name + "conv2d")
    if not use_bias:
        return _conv_out
    else:
        _conv_bias = flow.get_variable(
            name=name + "bias",
            shape=(out_channels,),
            initializer=flow.zeros_initializer(),
            dtype=x.dtype,
            trainable=trainable
        )
        return flow.nn.bias_add(_conv_out, _conv_bias, name=name + "_add_bias")


# Torch.nn.Linear default settings
def torch_style_linear(x, out_channels, trainable, name="dense"):
    """
    The Linear Layer.
    :param x: The input Tensor.
    :param out_channels: The output channels.
    :param trainable: Whether the dense weight is trained.
    :param name: The name for the Linear Layer.
    :return: The output Tensor.
    """
    _channel = x.shape[1]  # NCHW
    weight_shape = (out_channels, _channel)
    weight_regularizer = _get_regularizer("weight")
    # Kaiming Uniform initialize
    weight_init = flow.kaiming_initializer(
        shape=weight_shape, distribution="random_uniform", negative_slope=math.sqrt(5)
    )
    _dense_weight = flow.get_variable(
        name=name + "_dense_weight",
        shape=weight_shape,
        initializer=weight_init,
        regularizer=weight_regularizer,
        trainable=trainable
    )

    out = flow.matmul(x, _dense_weight, transpose_b=True, name=name + "_matmul")

    _fan_in = _channel  # The num of input channel
    bound = 1 / math.sqrt(_fan_in)
    bias_shape = (out_channels,)
    bias_init = flow.random_uniform_initializer(
        minval=-bound, maxval=bound, dtype=flow.float32
    )
    _bias = flow.get_variable(
        name=name + "_dense_bias",
        shape=bias_shape,
        initializer=bias_init,
        dtype=x.dtype,
        trainable=trainable
    )

    out = flow.nn.bias_add(out, _bias, name=name + "_add_bias")
    return out


def conv_bn(x, out_channels, kernel_size, stride, padding, groups=1, use_bias=False, trainable=True, training=True,
            name="conv_bn_"):
    """
    Build Conv+BN Layer
    :param x: The input Tensor.
    :param out_channels: The output channels.
    :param kernel_size: The kernelsize of ConvLayer.
    :param stride: The stride of ConvLayer.
    :param padding: The padding.
    :param groups: The groups of ConvLayer.
    :param use_bias: Whether use bias in ConvLayer.
    :param trainable: The param used in BNLayer.
    :param training: The param used in BNLayer.
    :param name: The name prefix for ConvLayer and BNLayer.
    :return: The output Tensor.
    """
    _conv_x = torch_style_conv(
        x, out_channels, kernel_size,
        stride, padding=[(0, 0), (0, 0), (padding, padding), (padding, padding)],
        groups=groups, use_bias=use_bias, trainable=trainable, name=name + "conv_layer_"
    )

    # The momentum and epsilon follow the pytorch default setting
    # Bn has no weight decay
    _bn_out = flow.layers.batch_normalization(
        _conv_x,
        momentum=0.9,
        epsilon=1e-5,
        axis=1,
        name=name + "bn_layer",
        trainable=trainable,
        training=training
    )
    return _bn_out


def repvggblock(x, out_channels, kernel_size, stride=1, padding=0,
                groups=1, deploy=False, trainable=True, training=True, name="repVGGBlock_"):
    """
    Build RepVGGBlock
    :param x: The input Tensor.
    :param out_channels: The output channels.
    :param kernel_size: The kernelsize of ConvLayer.
    :param stride: The stride of ConvLayer.
    :param padding: The padding.
    :param groups: The groups of ConvLayer.
    :param deploy: Whether to deploy. If deploy, the block only contains a 3x3 ConvLayer.
    :param trainable: The param used in BNLayer.
    :param training: The param used in BNLayer.
    :param name: The name for the Block.
    :return: Output Tensor.
    """
    assert kernel_size == 3
    assert padding == 1
    padding_1x1 = padding - kernel_size // 2

    if deploy:
        _reparam_padding = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        # Here conv need bias.
        _rbr_reparam = torch_style_conv(x, out_channels, kernel_size, stride, _reparam_padding, groups, True, trainable,
                                        name=name + "3x3_conv_layer_")
        return flow.math.relu(_rbr_reparam)
    else:
        # Here CONV+BN do not use bias, because BN has params `beta`.
        _use_bias = False

    in_channels = x.shape[1]
    _rbr_dense = conv_bn(
        x, out_channels, kernel_size, stride, padding, groups, _use_bias, trainable, training, name=name + "3x3_"
    )
    _rbr_1x1 = conv_bn(
        x, out_channels, kernel_size=1, stride=stride, padding=padding_1x1, groups=groups, use_bias=_use_bias,
        trainable=trainable, training=training, name=name + "1x1_"
    )
    if in_channels == out_channels and stride == 1:
        _rbr_identity = flow.layers.batch_normalization(
            x, momentum=0.9, axis=1, epsilon=1e-5, trainable=trainable, training=training,
            name=name + "identity_bn_layer"
        )
        return flow.math.relu(_rbr_identity + _rbr_dense + _rbr_1x1)
    else:
        return flow.math.relu(_rbr_dense + _rbr_1x1)


class RepVGG(object):
    def __init__(self, images, num_blocks, num_classes=1000, width_multiplier=None,
                 override_groups_map=None, trainable=True, training=True, deploy=False, name="RepVGG"):
        super(RepVGG, self).__init__()
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.images = images
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.trainable = trainable
        self.training = training
        self.deploy = deploy
        self.cur_layer_idx = 1
        self.name = name
        # For the first Block
        self.in_planes = min(64, int(64 * width_multiplier[0]))

    def _make_stage(self, x, planes, num_blocks, stride, name):
        """
        Make Block stage
        :param x: The input tensor
        :param planes: The num of filters
        :param num_blocks: The num of blocks
        :param stride: The stride
        :param name: The name
        :return:
        """
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            x = repvggblock(x, planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                            trainable=self.trainable, training=self.training,
                            name=name + "_" + str(self.cur_layer_idx) + "_")
            self.cur_layer_idx += 1
        return x

    def build_network(self):
        _stage0 = repvggblock(self.images, self.in_planes, kernel_size=3, stride=2, padding=1,
                              deploy=self.deploy, trainable=self.trainable, training=self.training,
                              name=self.name + "stage_0" + "_" + str(0) + "_")
        _stage1 = self._make_stage(_stage0, int(64 * self.width_multiplier[0]), num_blocks=self.num_blocks[0], stride=2,
                                   name=self.name + "stage_1")
        _stage2 = self._make_stage(_stage1, int(128 * self.width_multiplier[1]), num_blocks=self.num_blocks[1],
                                   stride=2,
                                   name=self.name + "stage_2")
        _stage3 = self._make_stage(_stage2, int(256 * self.width_multiplier[2]), num_blocks=self.num_blocks[2],
                                   stride=2,
                                   name=self.name + "stage_3")
        _stage4 = self._make_stage(_stage3, int(512 * self.width_multiplier[3]), num_blocks=self.num_blocks[3],
                                   stride=2,
                                   name=self.name + "stage_4")
        _gap = flow.nn.avg_pool2d(_stage4, ksize=[7, 7], strides=1, padding="VALID", name=self.name + "AveragePool")
        _flatten = flow.flatten(_gap, name=self.name + "flatten", start_dim=1, end_dim=-1)

        _linear = torch_style_linear(
            _flatten, self.num_classes, self.trainable, name=self.name + "classify"
        )

        return _linear


# The config of groups
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def RepVGG_A0(images, args, trainable=True, training=True, deploy=False):
    repvggA0 = RepVGG(images, num_blocks=[2, 4, 14, 1], num_classes=1000,
                      width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, trainable=trainable,
                      training=training, deploy=deploy)
    model = repvggA0.build_network()
    return model


def RepVGG_A1(images, args, trainable=True, training=True, deploy=False):
    repvggA1 = RepVGG(images, num_blocks=[2, 4, 14, 1], num_classes=1000,
                      width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                      trainable=trainable, training=training, deploy=deploy)
    model = repvggA1.build_network()
    return model


def RepVGG_A2(images, args, trainable=True, training=True, deploy=False):
    repvggA2 = RepVGG(images, num_blocks=[2, 4, 14, 1], num_classes=1000,
                      width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, trainable=trainable,
                      training=training, deploy=deploy)
    model = repvggA2.build_network()
    return model


def RepVGG_B0(images, args, trainable=True, training=True, deploy=False):
    repvggB0 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                      width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                      trainable=trainable, training=training, deploy=deploy)
    model = repvggB0.build_network()
    return model


def RepVGG_B1(images, args, trainable=True, training=True, deploy=False):
    repvggB1 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                      width_multiplier=[2, 2, 2, 4], override_groups_map=None,
                      trainable=trainable, training=training, deploy=deploy)
    model = repvggB1.build_network()
    return model


def RepVGG_B1g2(images, args, trainable=True, training=True, deploy=False):
    repvggB1g2 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB1g2.build_network()
    return model


def RepVGG_B1g4(images, args, trainable=True, training=True, deploy=False):
    repvggB1g4 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB1g4.build_network()
    return model


def RepVGG_B2(images, args, trainable=True, training=True, deploy=False):
    repvggB2 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                      width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                      trainable=trainable, training=training, deploy=deploy)
    model = repvggB2.build_network()
    return model


def RepVGG_B2g2(images, args, trainable=True, training=True, deploy=False):
    repvggB2g2 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB2g2.build_network()
    return model


def RepVGG_B2g4(images, args, trainable=True, training=True, deploy=False):
    repvggB2g4 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB2g4.build_network()
    return model


def RepVGG_B3(images, args, trainable=True, training=True, deploy=False):
    repvggB3 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                      width_multiplier=[3, 3, 3, 5], override_groups_map=None,
                      trainable=trainable, training=training, deploy=deploy)
    model = repvggB3.build_network()
    return model


def RepVGG_B3g2(images, args, trainable=True, training=True, deploy=False):
    repvggB3g2 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB3g2.build_network()
    return model


def RepVGG_B3g4(images, args, trainable=True, training=True, deploy=False):
    repvggB3g4 = RepVGG(images, num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map,
                        trainable=trainable, training=training, deploy=deploy)
    model = repvggB3g4.build_network()
    return model
