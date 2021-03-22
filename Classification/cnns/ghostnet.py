import oneflow as flow
import oneflow.nn as nn
import oneflow.typing as tp
import numpy as np
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _relu6(data):
     return flow.clip_by_value(data,0,6)

def hard_sigmoid(x, inplace: bool = False):
    # 实际上就是如果<=-1输出为0，>=1输出为1，中间为一个线性
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        #return F.relu6(x + 3.) / 6.
        return _relu6(x + 3.) / 6.

def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def _get_bias_initializer():
    return flow.zeros_initializer()

def conv2d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
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

    weight_shape = (filters, inputs.shape[1]//groups, kernel_size_1, kernel_size_2)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
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

    
# class ConvBnAct(nn.Module):
#     def __init__(self, in_chs, out_chs, kernel_size,
#                  stride=1, act_layer=nn.ReLU):
#         super(ConvBnAct, self).__init__()
#         self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_chs)
#         self.act1 = act_layer(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         return x
    
def conv2d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,    
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    use_bn=True,
):
    output = conv2d_layer(name=name,
                          inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          use_bias=use_bias,
                          groups=groups,
                          weight_initializer=weight_initializer,
                          bias_initializer=bias_initializer,
                          weight_regularizer=weight_regularizer,
                          bias_regularizer=bias_regularizer)

    if use_bn:
        output = flow.layers.batch_normalization(inputs=output,
                                                 axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=True,
                                                 name=name + "_bn")
    return output

def AdaptiveAvgPool2d(inputs, h_size, w_size):
    input_size = np.array(inputs.shape[2:])
    output_size = np.array([h_size,w_size])
    stride_size = np.floor(input_size/output_size).astype(np.int32)
    kernel_size = (input_size-(output_size-1)*stride_size)
    output = flow.nn.avg_pool2d(inputs, ksize=[int(kernel_size[0]),int(kernel_size[1])],
                                strides=[int(stride_size[0]),int(stride_size[1])], data_format="NCHW", padding="valid")
    return output

class SqueezeExcite(object):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        self.in_chs = in_chs
        self.reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        
    def build_network(self, inputs, times=1):
        output = AdaptiveAvgPool2d(inputs,1,1) 
        output = conv2d_layer("conv_se"+str(times)+"_0",output, self.reduced_chs, 1, use_bias=True)
        output = conv2d_layer("conv_se"+str(times)+"_1",output, self.in_chs, 1, activation=None, use_bias=True)
        output = self.gate_fn(output)
        return output

class GhostModule(object):
    def __init__(self, oup, ratio=2):
        super(GhostModule, self).__init__()
        self.oup = oup
        self.init_channels = math.ceil(oup / ratio) # math.ceil 返回大于等于参数x的最小整数
        self.new_channels = self.init_channels*(ratio-1) # 减少特征图

    def build_network(self, inputs, kernel_size=1, dw_size=3, stride=1, relu=True, times=1):
        output1 = conv2d_layer_with_bn("conv_gm"+str(times)+"_0", inputs, self.init_channels, kernel_size=kernel_size, 
                                       strides=stride, padding="same", activation=None, use_bias=False)
        if relu:
            output1 = flow.nn.relu(output1)

        # assert filters.shape[filter_in_axis] == inputs.shape[in_channel_axis] // groups
        output2 = conv2d_layer_with_bn("conv_gm"+str(times)+"_1", output1, self.new_channels, kernel_size=dw_size, strides=1, padding="same", groups=self.init_channels, activation=None, use_bias=False)
        if relu:
            output2 = flow.nn.relu(output2)
        output = flow.concat([output1,output2], axis=1)
        return output #[:,:self.oup,:,:]

class GhostBottleneck(object):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, ):
        super(GhostBottleneck, self).__init__()

    def build_network(self, inputs, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0., times=1):
        residual = inputs
        
        # Point-wise expansion
        ghost_model1 = GhostModule(mid_chs)
        output = ghost_model1.build_network(inputs, times=times)
        
        # Depth-wise convolution
        if stride > 1:
            output = conv2d_layer_with_bn("conv_gb"+str(times)+"_0", output, mid_chs, dw_kernel_size, strides=stride, padding="same",
                             activation=None, groups=mid_chs, use_bias=False)

        print(output.shape)
        # Squeeze-and-excitation
        if (se_ratio is not None and se_ratio > 0.):
            se_model = SqueezeExcite(mid_chs, se_ratio=se_ratio)
            output = se_model.build_network(output, times=times+1000)

        print(output.shape)
        # 2nd ghost bottleneck
        ghost_model2 = GhostModule(out_chs)
        output = ghost_model2.build_network(output, relu=False, times=times+1000)
        
        if (in_chs != out_chs or stride != 1):
            residual1 = conv2d_layer_with_bn("conv_gb"+str(times)+"_1",residual, in_chs, dw_kernel_size, strides=stride, 
                                            padding="SAME", activation=None, groups=in_chs, use_bias=False)
            residual = conv2d_layer_with_bn("conv_gb"+str(times)+"_2",residual1, out_chs, 1, strides=1, padding="valid", activation=None, use_bias=False)
          
        print(residual.shape)
        print(output.shape)
        output += residual
        return output

class GhostNet(object):
    def __init__(self, cfgs):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

    def build_network(self, inputs, num_classes=1000, width=1.0, dropout=0.2, trainable=True, wd=1.0 / 32768):
        kernel_initializer = flow.variance_scaling_initializer(2, 'fan_in', 'random_normal')
        weight_regularizer = flow.regularizers.l2(wd) if wd > 0.0 and wd < 1.0 else None
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        output = conv2d_layer_with_bn("conv_gn_0",inputs, output_channel, kernel_size=3, strides=2, #???
                                      padding="same", activation=None, use_bias=False)
        output = flow.nn.relu(output)
        input_channel = output_channel
        
        sum = 1
        model = GhostBottleneck()
        # building inverted residual blocks
        for cfg in self.cfgs:
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                # self, inputs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.
                output = model.build_network(output, input_channel, hidden_channel, output_channel, dw_kernel_size=k, stride=s, se_ratio=se_ratio, times=sum)
                input_channel = output_channel
                sum += 1
                
        output_channel = _make_divisible(exp_size * width, 4)
        output = conv2d_layer_with_bn("conv_gn_1",output, output_channel, 1, activation=None)
        #input_channel = output_channel
        
        # building last several layers
        output_channel = 1280
        output = AdaptiveAvgPool2d(output,1, 1)
        output = conv2d_layer("conv_gn_2",output, output_channel, 1, 1, padding="valid", use_bias=True)
        output = flow.nn.relu(output)
        print("***1")
        print(output.shape)
        output = flow.reshape(output,(output.shape[0],-1))
        print("***2")
        print(output.shape)
        if dropout > 0.:
            output = nn.dropout(output, rate=dropout)
        output = flow.layers.dense(output, 
                                   num_classes, 
                                   use_bias=True,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=flow.zeros_initializer(),
                                   kernel_regularizer=weight_regularizer,
                                   bias_regularizer=weight_regularizer,
                                   trainable=trainable)
        print("***3")
        print(output.shape)
        return output

def ghostnet(images, trainable=True, need_transpose=False, training=True, wd=1.0 / 32768, channel_last=False, **kwargs):
	"""
	Constructs a GhostNet model
	"""
	cfgs = [
		# k, t, c, SE, s 
		# stage1
		[[3,  16,  16, 0, 1]],
		# stage2
		[[3,  48,  24, 0, 2]],
		[[3,  72,  24, 0, 1]],
		# stage3
		[[5,  72,  40, 0.25, 2]],
		[[5, 120,  40, 0.25, 1]],
		# stage4
		[[3, 240,  80, 0, 2]],
		[[3, 200,  80, 0, 1],
		 [3, 184,  80, 0, 1],
		 [3, 184,  80, 0, 1],
		 [3, 480, 112, 0.25, 1],
		 [3, 672, 112, 0.25, 1]
		],
		# stage5
		[[5, 672, 160, 0.25, 2]],
		[[5, 960, 160, 0, 1],
		 [5, 960, 160, 0.25, 1],
		 [5, 960, 160, 0, 1],
		 [5, 960, 160, 0.25, 1]
		]
	]
	ghostnet = GhostNet(cfgs, **kwargs)
	model = ghostnet.build_network(images)
	return model
