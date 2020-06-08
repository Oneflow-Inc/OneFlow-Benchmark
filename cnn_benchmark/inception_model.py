from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow
from model_util import conv2d_layer_with_bn


def InceptionA(in_blob, index):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch5x5"):
            branch5x5_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=48, kernel_size=1, strides=1, padding="SAME"
            )
            branch5x5_2 = conv2d_layer_with_bn(
                "conv1",
                branch5x5_1,
                filters=64,
                kernel_size=5,
                strides=1,
                padding="SAME",
            )
        with flow.deprecated.variable_scope("branch3x3dbl"):
            branch3x3dbl_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = conv2d_layer_with_bn(
                "conv1",
                branch3x3dbl_1,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = conv2d_layer_with_bn(
                "conv2",
                branch3x3dbl_2,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
        with flow.deprecated.variable_scope("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = conv2d_layer_with_bn(
                "conv",
                branch_pool_1,
                filters=32 if index == 0 else 64,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )

        inceptionA_bn = []
        inceptionA_bn.append(branch1x1)
        inceptionA_bn.append(branch5x5_2)
        inceptionA_bn.append(branch3x3dbl_3)
        inceptionA_bn.append(branch_pool_2)

        mixed_concat = flow.concat(values=inceptionA_bn, axis=1, name="concat")

    return mixed_concat


def InceptionB(in_blob, index):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch3x3"):
            branch3x3 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=384, kernel_size=3, strides=2, padding="VALID"
            )
        with flow.deprecated.variable_scope("branch3x3dbl"):
            branch3x3dbl_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=64, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = conv2d_layer_with_bn(
                "conv1",
                branch3x3dbl_1,
                filters=96,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = conv2d_layer_with_bn(
                "conv2",
                branch3x3dbl_2,
                filters=96,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.deprecated.variable_scope("branch_pool"):
            branch_pool = flow.nn.max_pool2d(
                in_blob,
                ksize=3,
                strides=2,
                padding="VALID",
                data_format="NCHW",
                name="pool0",
            )

        inceptionB_bn = []
        inceptionB_bn.append(branch3x3)
        inceptionB_bn.append(branch3x3dbl_3)
        inceptionB_bn.append(branch_pool)
        mixed_concat = flow.concat(values=inceptionB_bn, axis=1, name="concat")

    return mixed_concat


def InceptionC(in_blob, index, filters):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch7x7"):
            branch7x7_1 = conv2d_layer_with_bn(
                "conv0",
                in_blob,
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )
            branch7x7_2 = conv2d_layer_with_bn(
                "conv1",
                branch7x7_1,
                filters=filters,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7_3 = conv2d_layer_with_bn(
                "conv2",
                branch7x7_2,
                filters=192,
                kernel_size=[7, 1],
                strides=[1, 1],
                padding="SAME",
            )
        with flow.deprecated.variable_scope("branch7x7dbl"):
            branch7x7dbl_1 = conv2d_layer_with_bn(
                "conv0",
                in_blob,
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_2 = conv2d_layer_with_bn(
                "conv1",
                branch7x7dbl_1,
                filters=filters,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_3 = conv2d_layer_with_bn(
                "conv2",
                branch7x7dbl_2,
                filters=filters,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_4 = conv2d_layer_with_bn(
                "conv3",
                branch7x7dbl_3,
                filters=filters,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7dbl_5 = conv2d_layer_with_bn(
                "conv4",
                branch7x7dbl_4,
                filters=192,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
        with flow.deprecated.variable_scope("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = conv2d_layer_with_bn(
                "conv",
                branch_pool_1,
                filters=192,
                kernel_size=[1, 1],
                strides=1,
                padding="SAME",
            )

        inceptionC_bn = []
        inceptionC_bn.append(branch1x1)
        inceptionC_bn.append(branch7x7_3)
        inceptionC_bn.append(branch7x7dbl_5)
        inceptionC_bn.append(branch_pool_2)
        mixed_concat = flow.concat(values=inceptionC_bn, axis=1, name="concat")

    return mixed_concat


def InceptionD(in_blob, index):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch3x3"):
            branch3x3_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3_2 = conv2d_layer_with_bn(
                "conv1",
                branch3x3_1,
                filters=320,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.deprecated.variable_scope("branch7x7x3"):
            branch7x7x3_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=192, kernel_size=1, strides=1, padding="SAME"
            )
            branch7x7x3_2 = conv2d_layer_with_bn(
                "conv1",
                branch7x7x3_1,
                filters=192,
                kernel_size=[1, 7],
                strides=1,
                padding="SAME",
            )
            branch7x7x3_3 = conv2d_layer_with_bn(
                "conv2",
                branch7x7x3_2,
                filters=192,
                kernel_size=[7, 1],
                strides=1,
                padding="SAME",
            )
            branch7x7x3_4 = conv2d_layer_with_bn(
                "conv3",
                branch7x7x3_3,
                filters=192,
                kernel_size=3,
                strides=2,
                padding="VALID",
            )
        with flow.deprecated.variable_scope("branch_pool"):
            branch_pool = flow.nn.max_pool2d(
                in_blob,
                ksize=3,
                strides=2,
                padding="VALID",
                data_format="NCHW",
                name="pool",
            )

        inceptionD_bn = []
        inceptionD_bn.append(branch3x3_2)
        inceptionD_bn.append(branch7x7x3_4)
        inceptionD_bn.append(branch_pool)

        mixed_concat = flow.concat(values=inceptionD_bn, axis=1, name="concat")

    return mixed_concat


def InceptionE(in_blob, index):
    with flow.deprecated.variable_scope("mixed_{}".format(index)):
        with flow.deprecated.variable_scope("branch1x1"):
            branch1x1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=320, kernel_size=1, strides=1, padding="SAME"
            )
        with flow.deprecated.variable_scope("branch3x3"):
            branch3x3_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=384, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3_2 = conv2d_layer_with_bn(
                "conv1",
                branch3x3_1,
                filters=384,
                kernel_size=[1, 3],
                strides=1,
                padding="SAME",
            )
            branch3x3_3 = conv2d_layer_with_bn(
                "conv2",
                branch3x3_1,
                filters=384,
                kernel_size=[3, 1],
                strides=[1, 1],
                padding="SAME",
            )
            inceptionE_1_bn = []
            inceptionE_1_bn.append(branch3x3_2)
            inceptionE_1_bn.append(branch3x3_3)
            concat_branch3x3 = flow.concat(
                values=inceptionE_1_bn, axis=1, name="concat"
            )
        with flow.deprecated.variable_scope("branch3x3dbl"):
            branch3x3dbl_1 = conv2d_layer_with_bn(
                "conv0", in_blob, filters=448, kernel_size=1, strides=1, padding="SAME"
            )
            branch3x3dbl_2 = conv2d_layer_with_bn(
                "conv1",
                branch3x3dbl_1,
                filters=384,
                kernel_size=3,
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_3 = conv2d_layer_with_bn(
                "conv2",
                branch3x3dbl_2,
                filters=384,
                kernel_size=[1, 3],
                strides=1,
                padding="SAME",
            )
            branch3x3dbl_4 = conv2d_layer_with_bn(
                "conv3",
                branch3x3dbl_2,
                filters=384,
                kernel_size=[3, 1],
                strides=1,
                padding="SAME",
            )
            inceptionE_2_bn = []
            inceptionE_2_bn.append(branch3x3dbl_3)
            inceptionE_2_bn.append(branch3x3dbl_4)
            concat_branch3x3dbl = flow.concat(
                values=inceptionE_2_bn, axis=1, name="concat"
            )
        with flow.deprecated.variable_scope("branch_pool"):
            branch_pool_1 = flow.nn.avg_pool2d(
                in_blob,
                ksize=3,
                strides=1,
                padding="SAME",
                data_format="NCHW",
                name="pool",
            )
            branch_pool_2 = conv2d_layer_with_bn(
                "conv",
                branch_pool_1,
                filters=192,
                kernel_size=[1, 1],
                strides=1,
                padding="SAME",
            )

        inceptionE_total_bn = []
        inceptionE_total_bn.append(branch1x1)
        inceptionE_total_bn.append(concat_branch3x3)
        inceptionE_total_bn.append(concat_branch3x3dbl)
        inceptionE_total_bn.append(branch_pool_2)

        concat_total = flow.concat(
            values=inceptionE_total_bn, axis=1, name="concat")

    return concat_total


def inceptionv3(images, trainable=True, need_transpose=False):

    if need_transpose:
        images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])

    with flow.deprecated.variable_scope("InceptionV3"):
        # conv0: 299 x 299 x 3
        conv0 = conv2d_layer_with_bn(
            "conv0", images, filters=32, kernel_size=3, strides=2, padding="VALID"
        )
        conv1 = conv2d_layer_with_bn(
            "conv1", conv0, filters=32, kernel_size=3, strides=1, padding="VALID"
        )
        conv2 = conv2d_layer_with_bn(
            "conv2", conv1, filters=64, kernel_size=3, strides=1, padding="SAME"
        )
        pool1 = flow.nn.max_pool2d(
            conv2, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool1"
        )
        conv3 = conv2d_layer_with_bn(
            "conv3", pool1, filters=80, kernel_size=1, strides=1, padding="VALID"
        )
        conv4 = conv2d_layer_with_bn(
            "conv4", conv3, filters=192, kernel_size=3, strides=1, padding="VALID"
        )
        pool2 = flow.nn.max_pool2d(
            conv4, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool2"
        )

        # mixed_0 ~ mixed_2
        mixed_0 = InceptionA(pool2, 0)
        mixed_1 = InceptionA(mixed_0, 1)
        mixed_2 = InceptionA(mixed_1, 2)
        # mixed_3
        mixed_3 = InceptionB(mixed_2, 3)

        # mixed_4 ~ mixed_7
        mixed_4 = InceptionC(mixed_3, 4, 128)
        mixed_5 = InceptionC(mixed_4, 5, 160)
        mixed_6 = InceptionC(mixed_5, 6, 160)
        mixed_7 = InceptionC(mixed_6, 7, 192)

        # mixed_8
        mixed_8 = InceptionD(mixed_7, 8)

        # mixed_9 ~ mixed_10
        mixed_9 = InceptionE(mixed_8, 9)
        mixed_10 = InceptionE(mixed_9, 10)

        pool3 = flow.nn.avg_pool2d(
            mixed_10, ksize=8, strides=1, padding="VALID", data_format="NCHW", name="pool3"
        )

        # TODO: Need to transpose weight when converting model from TF to OF if
        # you want to use layers.dense interface.
        fc1 = flow.layers.dense(
            inputs=flow.reshape(pool3, [pool3.shape[0], -1]),
            units=1001,
            activation=None,
            use_bias=True,
            kernel_initializer=flow.truncated_normal(0.816496580927726),
            bias_initializer=flow.constant_initializer(),
            trainable=trainable,
            name="fc1",
        )

    return fc1
