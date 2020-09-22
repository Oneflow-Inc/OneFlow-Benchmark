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
#ref : https://arxiv.org/pdf/1801.04381.pdf
#ref : https://github.com/liangfu/mxnet-mobilenet-v2/blob/master/symbols/mobilenetv2.py

def _get_regularizer(model_name):
    #all decay
    return flow.regularizers.l2(0.00004)


def _get_initializer(model_name):
    if model_name == "weight":
        return flow.variance_scaling_initializer(2.0, mode="fan_out", distribution="random_normal", data_format="NCHW")
    elif model_name == "bias":
        return flow.zeros_initializer()
    elif model_name == "gamma":
        return flow.ones_initializer()
    elif model_name == "beta":
        return flow.zeros_initializer()
    elif model_name == "dense_weight":
        return flow.random_normal_initializer(0, 0.01)


def _batch_norm(inputs, axis, momentum, epsilon, center=True, scale=True, trainable=True, name=None):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer = _get_initializer("beta"),
        gamma_initializer = _get_initializer("gamma"),
        beta_regularizer = _get_regularizer("beta"),
        gamma_regularizer = _get_regularizer("gamma"),
        trainable=trainable,
        name=name
    )


def _relu6(data, prefix):
    return flow.clip_by_value(data,0,6,name='%s-relu6'%prefix)


def mobilenet_unit(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, data_format="NCHW", if_act=True, use_bias=False, prefix=''):
    conv = flow.layers.conv2d(inputs=data, filters=num_filter, kernel_size=kernel, strides=stride, 
            padding=pad, data_format=data_format, dilation_rate=1, groups=num_group, activation=None, 
            use_bias=use_bias, kernel_initializer=_get_initializer("weight"), 
            bias_initializer=_get_initializer("bias"), kernel_regularizer=_get_regularizer("weight"), 
            bias_regularizer=_get_regularizer("bias"), name=prefix)
    bn = _batch_norm(conv, axis=1, momentum=0.9, epsilon=1e-5, name='%s-BatchNorm'%prefix)
    if if_act:
        act = _relu6(bn, prefix)
        return act
    else:
        return bn

def shortcut(data_in, data_residual, prefix):
    out = flow.math.add(data_in,data_residual)
    return out

def inverted_residual_unit(data, num_in_filter, num_filter, ifshortcut, stride, kernel, pad, expansion_factor, prefix, data_format="NCHW", has_expand = 1):
    num_expfilter = int(round(num_in_filter*expansion_factor))
    if has_expand:
        channel_expand = mobilenet_unit(
            data=data,
            num_filter=num_expfilter,
            kernel=(1,1),
            stride=(1,1),
            pad="valid",
            num_group=1,
            data_format=data_format,
            if_act=True,
            prefix='%s-expand'%prefix,
        )
    else:
        channel_expand = data
    bottleneck_conv = mobilenet_unit(
        data= channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        data_format=data_format,
        if_act=True,
        prefix='%s-depthwise'%prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad="valid",
        num_group=1,
        data_format=data_format,
        if_act=False,
        prefix='%s-project'%prefix
    )

    if ifshortcut:
        out = shortcut(
            data_in=data,
            data_residual=linear_out,
            prefix=prefix,
        ) 
        return out
    else:
        return linear_out

MNETV2_CONFIGS_MAP = {
    (224,224):{
        'firstconv_filter_num': 32, 
        # t, c, s
        'bottleneck_params_list':[
            (1, 16, 1, False), 
            (6, 24, 2, False), 
            (6, 24, 1, True), 
            (6, 32, 2, False), 
            (6, 32, 1, True), 
            (6, 32, 1, True), 
            (6, 64, 2, False), 
            (6, 64, 1, True), 
            (6, 64, 1, True), 
            (6, 64, 1, True), 
            (6, 96, 1, False), 
            (6, 96, 1, True), 
            (6, 96, 1, True), 
            (6, 160, 2, False), 
            (6, 160, 1, True), 
            (6, 160, 1, True), 
            (6, 320, 1, False), 
        ],
        'filter_num_before_gp': 1280, 
    } 
}

class MobileNetV2(object):
    def __init__(self, data_wh, multiplier, **kargs):
        super(MobileNetV2, self).__init__()
        self.data_wh=data_wh
        self.multiplier=multiplier
        if self.data_wh in MNETV2_CONFIGS_MAP:
            self.config_map=MNETV2_CONFIGS_MAP[self.data_wh]
        else:
            self.config_map=MNETV2_CONFIGS_MAP[(224, 224)]
    
    def build_network(self, input_data, data_format, class_num=1000, prefix="", **configs):
        self.config_map.update(configs)

        first_c = int(round(self.config_map['firstconv_filter_num']*self.multiplier))
        first_layer = mobilenet_unit(
            data=input_data,
            num_filter=first_c,
            kernel=(3,3),
            stride=(2,2),
            pad="same",
            data_format=data_format,
            if_act=True,
            prefix= prefix+'-Conv'
        )

        last_bottleneck_layer = first_layer
        in_c = first_c
        for i, layer_setting in enumerate(self.config_map['bottleneck_params_list']):
            t, c, s, sc = layer_setting
            if i == 0:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c*self.multiplier)),
                    ifshortcut=sc,
                    stride=(s,s),
                    kernel=(3,3),
                    pad="same",
                    expansion_factor=t,
                    prefix= prefix+'-expanded_conv',
                    data_format=data_format,
                    has_expand=0
                )
                in_c = int(round(c*self.multiplier))
            else:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c*self.multiplier)),
                    ifshortcut=sc,
                    stride=(s,s),
                    kernel=(3,3),
                    pad="same",
                    expansion_factor=t,
                    data_format=data_format,
                    prefix= prefix+'-expanded_conv_%d'%i
                )
                in_c = int(round(c*self.multiplier))
        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            num_filter=int(1280 * self.multiplier) if self.multiplier > 1.0 else 1280,
            kernel=(1,1),
            stride=(1,1),
            pad="valid",
            data_format=data_format,
            if_act=True,
            prefix=prefix+'-Conv_1'
        )
        # global average pooling
        pool_size = int(self.data_wh[0] / 32)  
        pool = flow.nn.avg_pool2d(
            last_fm, ksize=pool_size, strides=1, padding="VALID", data_format="NCHW", name="pool5",
        ) 
        fc = flow.layers.dense(
            flow.reshape(pool, (pool.shape[0], -1)),
            units=class_num,
            use_bias=False,
            kernel_initializer=_get_initializer("dense_weight"),
            bias_initializer=_get_initializer("bias"),
            kernel_regularizer=_get_regularizer("dense_weight"),
            bias_regularizer=_get_regularizer("bias"),
            name=prefix+'-fc',
        )
        return fc

    def __call__(self, input_data, class_num=1000, prefix = "", **configs):
        sym = self.build_network(input_data, class_num=class_num, prefix=prefix, **configs)
        return sym

def Mobilenet(input_data, args, trainable=True, training=True, num_classes=1000, multiplier=1.0, prefix = ""):
    assert   args.channel_last==False, "Mobilenet does not support channel_last mode, set channel_last=False will be right!"
    data_format="NHWC" if args.channel_last else "NCHW"
    mobilenetgen = MobileNetV2((224,224), multiplier=multiplier)
    out = mobilenetgen(input_data, data_format=data_format, class_num=num_classes, prefix = "MobilenetV2")
    return out
