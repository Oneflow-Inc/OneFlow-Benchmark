import oneflow as flow
import oneflow.python.framework.distribute as distribute_util

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


# def batchnorm(input, name, axis=1, reuse=False):
#     name_ = name if reuse == False else name + "_reuse"
#     return flow.layers.batch_normalization(input, axis=axis, name=name_)

def batchnorm(
        inputs,
        axis=1,
        momentum: float = 0.99,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        beta_initializer=None,
        gamma_initializer=None,
        beta_regularizer=None,
        gamma_regularizer=None,
        moving_mean_initializer=None,
        moving_variance_initializer=None,
        trainable=True,
        training=True,
        name: str = "BatchNorm",
        reuse=False,
    ):
        name_ = name if reuse == False else name + "_reuse"
        if axis < 0:
            axis += len(inputs.shape)
        assert axis >= 0 and axis < len(inputs.shape)

        params_shape = [inputs.shape[axis]]
        # Float32 required to avoid precision-loss when using fp16 input/output
        params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

        if not flow.current_global_function_desc().IsTrainable() or not trainable:
            training = False

        if center:
            beta = flow.get_variable(
                name=name + "beta",
                shape=params_shape,
                dtype=params_dtype,
                initializer=beta_initializer or flow.zeros_initializer(),
                regularizer=beta_regularizer,
                trainable=trainable,
                distribute=distribute_util.broadcast(),
                reuse=reuse,
            )
        else:
            beta = flow.constant(0, dtype=params_dtype, shape=params_shape, name="beta")

        if scale:
            gamma = flow.get_variable(
                name=name + "gamma",
                shape=params_shape,
                dtype=params_dtype,
                initializer=gamma_initializer or flow.ones_initializer(),
                regularizer=gamma_regularizer,
                trainable=trainable,
                distribute=distribute_util.broadcast(),
                reuse=reuse,
            )
        else:
            gamma = flow.constant(
                1, dtype=params_dtype, shape=params_shape, name="gamma"
            )

        moving_mean = flow.get_variable(
            name=name + "moving_mean",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_mean_initializer or flow.zeros_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
            reuse=reuse,
        )

        moving_variance = flow.get_variable(
            name=name + "moving_variance",
            shape=params_shape,
            dtype=params_dtype,
            initializer=moving_variance_initializer or flow.ones_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
            reuse=reuse,
        )

        builder = (
            flow.user_op_builder(name_)
            .Op("normalization")
            .Input("x", [inputs])
            .Input("moving_mean", [moving_mean])
            .Input("moving_variance", [moving_variance])
            .Input("gamma", [gamma])
            .Input("beta", [beta])
            .Output("y")
            .Attr("axis", axis)
            .Attr("epsilon", epsilon)
            .Attr("training", training)
            .Attr("momentum", momentum)
        )
        if trainable and training:
            builder = builder.Output("mean").Output("inv_variance")

        return builder.Build().InferAndTryRun().RemoteBlobList()[0]

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
