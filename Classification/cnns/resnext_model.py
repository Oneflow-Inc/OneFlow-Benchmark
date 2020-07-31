from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow
import collections


__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

basic_block_expansion = 1
bottle_neck_expansion = 4


def _conv2d(
    inputs,
    filters,
    kernel_size,
    strides=1,
    padding="VALID",
    groups=1,
    use_bias=False,
    trainable=True,
    name=None
):
    if padding != "SAME" and padding != "VALID":
        if isinstance(padding, list):
            inputs = flow.pad(inputs, (padding))
            padding = "VALID"
        elif isinstance(padding, tuple):
            inputs = flow.pad(inputs, padding)
            padding = "VALID"
        else:
            raise ValueError("padding must be SAME， VALID or a list/tuple.")

    return flow.layers.conv2d(
        inputs, filters, kernel_size, strides, padding,
        data_format="NCHW", dilation_rate=1, groups=groups,
        activation=None, use_bias=use_bias,
        kernel_initializer=flow.random_normal_initializer(),
        bias_initializer=flow.zeros_initializer(),
        kernel_regularizer=None, bias_regularizer=None,
        trainable=trainable, name=name, weight_name=name+"-weight",
        bias_name=name+"-bias")

def conv3x3(in_tensor, filters, strides=1, groups=1, trainable=True, name=""):
    return _conv2d(in_tensor, filters=filters, kernel_size=3,
            strides=strides, padding=([0, 0], [0, 0], [1, 1], [1, 1]), groups=groups, use_bias=False,
            trainable=trainable, name=name)


def _batch_norm(inputs, trainable=True, training=True, name=None):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.1,
        epsilon=1e-5,
        center=True,
        scale=True,
        trainable=trainable,
        training=training,
        name=name
    )


def basic_block(inputs, filters, strides=1, downsample=None, num_group=32,
        trainable=True, training=True, layer_block=""):
    residual = inputs
    conv1 = conv3x3(inputs, filters*2, strides, trainable=trainable, name=layer_block+"conv1")
    bn1 = _batch_norm(conv1, trainable=trainable, training=training, name=layer_block+"bn1")
    relu = flow.nn.relu(bn1, name=layer_block+"relu1")
    conv2 = conv3x3(relu, filters*2, groups=num_group, trainable=trainable,
            name=layer_block+"conv2")
    bn2 = _batch_norm(conv2, trainable=trainable, training=training, name=layer_block+"bn2")
    if downsample is True:
        residual =  _conv2d(inputs, filters * basic_block_expansion,
                          kernel_size=1, strides=strides, use_bias=False,
                          trainable=trainable, name=layer_block+"downsample-0")
        residual = _batch_norm(residual, trainable, training=training, name=layer_block+"downsampe-1")
    out = flow.math.add(bn2, residual)
    out = flow.nn.relu(out)
    return out


def bottle_neck(inputs, filters, strides,
        downsample=None, num_group=32, trainable=True, training=True, layer_block=""):
    residual = inputs
    conv1 = _conv2d(inputs, filters*2, kernel_size=1, trainable=trainable, name=layer_block+"conv1")
    bn1 = _batch_norm(conv1, trainable=trainable, training=training, name=layer_block+"bn1")
    relu1 = flow.nn.relu(bn1, name=layer_block+"relu1")
    conv2 = _conv2d(relu1, filters*2, kernel_size=3, strides=strides,
            padding=([0, 0], [0, 0], [1, 1], [1, 1]), use_bias=False,
            groups=num_group, trainable=trainable,
            name=layer_block+"conv2")
    bn2 = _batch_norm(conv2, trainable=trainable, training=training, name=layer_block+"bn2")
    relu2 = flow.nn.relu(bn2, name=layer_block+"relu2")
    conv3 = _conv2d(relu2, filters*4, kernel_size=1, padding="VALID",
            use_bias=False, trainable=trainable, name=layer_block+"conv3")
    bn3 = _batch_norm(conv3, training=training, name=layer_block+"bn3") # pass
    if downsample is True:
        residual  =  _conv2d(inputs, filters * bottle_neck_expansion,
                          kernel_size=1, strides=strides, use_bias=False,
                          trainable=trainable,
                          name=layer_block+"downsample-0")
        residual  = _batch_norm(residual, trainable=trainable,
                training=training, name=layer_block+"downsample-1")
    out = flow.math.add(bn3, residual)
    out = flow.nn.relu(out)

    return out


class ResNeXt():
    def __init__(self, images, trainable=True, training=True,
        need_transpose=False, channel_last=False, block=None, layers=[],
        num_classes=1000, num_group=32):
        self.input = 64
        self.images = images
        self.trainable = trainable
        self.training = training
        self.data_format = "NHWC" if channel_last else "NCHW"
        self.need_transpose=need_transpose
        self.layers = layers
        self.block = block
        self.num_classes = num_classes
        self.num_group = num_group
        self.block_expansion = 1 if self.block == basic_block else 4
        super(ResNeXt, self).__init__()


    def _make_layer(self, inputs, filters, blocks, num_group, strides=1,
            layer_num=""):
        downsample = None
        if strides != 1 or self.input != filters * self.block_expansion:
            downsample = True
        block_out = self.block(inputs, filters, strides,
                downsample, num_group=self.num_group, trainable=self.trainable,
                training=self.training,
             layer_block=layer_num+"-0-")

        layers = []
        layers.append(block_out)
        self.input = filters * self.block_expansion
        for i in range(1, blocks):
            block_out = self.block(block_out, filters,
                    strides=1, downsample=False, num_group=num_group,
                    trainable=self.trainable, training=self.training,
                    layer_block=layer_num+"-"+str(i)+"-")
            layers.append(block_out)
        return layers

    def build_network(self):
        if self.need_transpose:
            images = flow.transpose(self.images, name="transpose", perm=[0, 3, 1,
            2])
        else:
            images = self.images
        conv1 = _conv2d(images, 64, kernel_size=7, strides=2,
                padding=([0, 0], [0, 0], [3, 3], [3, 3]),
                groups=1, use_bias=False, trainable=self.trainable, name="conv1")

        bn1  = _batch_norm(conv1, trainable=self.trainable, training=self.training, name="bn1")

        relu = flow.nn.relu(bn1, name="relu1")
        pad_before_max_pool = flow.pad(relu, ([0, 0], [0, 0], [1, 1], [1, 1]))
        max_pool = flow.nn.max_pool2d(relu, ksize=3, strides=2,
                padding="VALID", data_format="NCHW", name="max_pool")  
        layer1 = self._make_layer(max_pool, 64, self.layers[0],
                self.num_group, layer_num="layer1")
        layer2 = self._make_layer(layer1[-1], 128, self.layers[1],
                self.num_group, strides=2, layer_num="layer2")
        layer3 = self._make_layer(layer2[-1], 256, self.layers[2],
                self.num_group, strides=2, layer_num="layer3")
        layer4 = self._make_layer(layer3[-1], 512, self.layers[3],
                self.num_group, strides=2, layer_num="layer4")

        # debug mode: dump data for debugging
        # with flow.watch_scope(blob_watcher=blob_watched,
        #    diff_blob_watcher=diff_blob_watched):
        #    bn1_identity = flow.identity(layer4[-1], name="layer4_last_out")

        avg_pool = flow.nn.avg_pool2d(layer4[-1], 7, strides=1, padding="VALID",
                data_format="NCHW", name="avg_pool")

        reshape = flow.reshape(avg_pool, (avg_pool.shape[0], -1))

        fc = flow.layers.dense(reshape, units=self.num_classes, use_bias=True,
                kernel_initializer=flow.xavier_uniform_initializer(),
                bias_initializer=flow.zeros_initializer(),
                trainable=self.trainable,
                name="fc")
        return fc


def resnext18(images, trainable=True, training=True, need_transpose=False,
        channel_last=False, **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    resnext_18 = ResNeXt(images, trainable=trainable, training=training,
            need_transpose=need_transpose, channel_last=channel_last,
            block=basic_block, layers=[2, 2, 2, 2], **kwargs)
    model = resnext_18.build_network()
    return model


def resnext34(images, trainable=True, training=True, need_transpose=False,
        channel_last=False, **kwargs):
    """Constructs a ResNeXt-34 model.
    """
    resnext_34 = ResNeXt(images, trainable=trainable, training=training,
            need_transpose=False, channel_last=False,
            block=basic_block, layers=[3, 4, 6, 3], **kwargs)
    model = resnext_34.build_network()
    return model


def resnext50(images, trainable=True, training=True, need_transpose=False,
        channel_last=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    """
    resnext_50 = ResNeXt(images,  trainable=trainable, training=training,
             need_transpose=need_transpose, channel_last=channel_last,
             block=bottle_neck, layers=[3, 4, 6, 3], **kwargs)
    model = resnext_50.build_network()
    return model


def resnext101(images, trainable=True, training=True, need_transpose=False,
        channel_last=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    resnext_101 = ResNeXt(images, trainable=trainable, training=training,
            need_transpose=False, channel_last=False,
            block=bottle_neck, layers=[3, 4, 23, 3], **kwargs)
    model = resnex_101.build_network()
    return model


def resnext152(images, trainable=True, training=True, need_transpose=False,
        channel_last=False, **kwargs):
    """Constructs a ResNeXt-152 model.
    """
    resnext_152 = ResNeXt(images, trainable=trainable, training=training,
            need_transpose=need_transpose, channel_last=channel_last,
            block=bottle_neck, layers=[3, 8, 36, 3], **kwargs)
    model = resnext_152.build_network()
    return model
