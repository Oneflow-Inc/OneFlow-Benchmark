import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np

# global vars
OUTPUT_CHANNELS = 3
LAMBDA = 100
BATCH_SIZE = 3

# download dataset
def download():
    # the default download path is "~/.keras/datasets"
    _URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
    path_to_zip = tf.keras.utils.get_file("facades.tar.gz", origin=_URL, extract=True)
    return path_to_zip


# load data, which is a single image with shape (256, 256, 3)
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# build the model
def downsample(inp, filters, size, apply_batchnorm=True, const_init=True):
    if not const_init:
        initializer = tf.random_normal_initializer(0.0, 0.02)
    else:
        initializer = tf.constant_initializer(0.002)

    conv = tf.keras.layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
    )(inp)

    if apply_batchnorm:
        conv = tf.keras.layers.BatchNormalization()(conv)

    result = tf.keras.layers.LeakyReLU()(conv)

    return result


def upsample(inp, filters, size, apply_dropout=False, const_init=True):
    if not const_init:
        initializer = tf.random_normal_initializer(0.0, 0.02)
    else:
        initializer = tf.constant_initializer(0.002)

    deconv = tf.keras.layers.Conv2DTranspose(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
    )(inp)

    deconv = tf.keras.layers.BatchNormalization()(deconv)

    if apply_dropout:
        deconv = tf.keras.layers.Dropout(0.5)(deconv)

    result = tf.keras.layers.ReLU()(deconv)

    return result


def Generator(const_init=True):
    apply_dropout = False if const_init else True

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    d1 = downsample(
        inputs, 64, 4, apply_batchnorm=False, const_init=const_init
    )  # (bs, 128, 128, 64)
    d2 = downsample(d1, 128, 4, const_init=const_init)  # (bs, 64, 64, 128)
    d3 = downsample(d2, 256, 4, const_init=const_init)  # (bs, 32, 32, 256)
    d4 = downsample(d3, 512, 4, const_init=const_init)  # (bs, 16, 16, 512)
    d5 = downsample(d4, 512, 4, const_init=const_init)  # (bs, 8, 8, 512)
    d6 = downsample(d5, 512, 4, const_init=const_init)  # (bs, 4, 4, 512)
    d7 = downsample(d6, 512, 4, const_init=const_init)  # (bs, 2, 2, 512)
    d8 = downsample(d7, 512, 4, const_init=const_init)  # (bs, 1, 1, 512)

    u7 = upsample(
        d8, 512, 4, apply_dropout=apply_dropout, const_init=const_init
    )  # (bs, 2, 2, 1024)
    u7 = tf.keras.layers.Concatenate()([u7, d7])
    u6 = upsample(
        u7, 512, 4, apply_dropout=apply_dropout, const_init=const_init
    )  # (bs, 4, 4, 1024)
    u6 = tf.keras.layers.Concatenate()([u6, d6])
    u5 = upsample(
        u6, 512, 4, apply_dropout=apply_dropout, const_init=const_init
    )  # (bs, 8, 8, 1024)
    u5 = tf.keras.layers.Concatenate()([u5, d5])
    u4 = upsample(u5, 512, 4, const_init=const_init)  # (bs, 16, 16, 1024)
    u4 = tf.keras.layers.Concatenate()([u4, d4])
    u3 = upsample(u4, 256, 4, const_init=const_init)  # (bs, 32, 32, 512)
    u3 = tf.keras.layers.Concatenate()([u3, d3])
    u2 = upsample(u3, 128, 4, const_init=const_init)  # (bs, 64, 64, 256)
    u2 = tf.keras.layers.Concatenate()([u2, d2])
    u1 = upsample(u2, 64, 4, const_init=const_init)  # (bs, 128, 128, 128)
    u1 = tf.keras.layers.Concatenate()([u1, d1])

    if not const_init:
        initializer = tf.random_normal_initializer(0.0, 0.02)
    else:
        initializer = tf.constant_initializer(0.002)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (bs, 256, 256, 3)
    u0 = last(u1)

    return tf.keras.Model(inputs=inputs, outputs=u0)


def Discriminator(const_init=True):
    if not const_init:
        initializer = tf.random_normal_initializer(0.0, 0.02)
    else:
        initializer = tf.constant_initializer(0.002)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(
        x, 64, 4, apply_batchnorm=False, const_init=const_init
    )  # (bs, 128, 128, 64)
    down2 = downsample(down1, 128, 4, const_init=const_init)  # (bs, 64, 64, 128)
    down3 = downsample(down2, 256, 4, const_init=const_init)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv, training=True)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def tf_pix2pix_test():
    inp = tf.random.normal([BATCH_SIZE, 256, 256, 3])
    tar = tf.random.normal([BATCH_SIZE, 256, 256, 3])

    np.save("input.npy", inp)
    np.save("target.npy", tar)

    generator = Generator()
    result = generator(inp, training=True)
    discriminator = Discriminator()
    result = discriminator([result, tar], training=True)
    print(result.shape)
    np.save("result.npy", result.numpy())
