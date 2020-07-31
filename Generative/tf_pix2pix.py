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
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np

# global vars
OUTPUT_CHANNELS = 3
LAMBDA = 100
BATCH_SIZE = 1

# download dataset

def get_constant_initializer(constant_value=0.002):
    return tf.constant_initializer(constant_value)

def download():
    # the default download path is "~/.keras/datasets"
    _URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
    path_to_zip = tf.keras.utils.get_file(
        "facades.tar.gz", origin=_URL, extract=True)
    return path_to_zip


# build the model
def downsample(inp, filters, size, apply_batchnorm=True, const_init=True):
    conv = tf.keras.layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=get_constant_initializer(),
        use_bias=False,
    )(inp)

    if apply_batchnorm: #and not const_init:
        conv = tf.keras.layers.BatchNormalization()(conv)

    result = tf.keras.layers.LeakyReLU()(conv)

    return result


def upsample(inp, filters, size, apply_dropout=False, const_init=True):
    deconv = tf.keras.layers.Conv2DTranspose(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=get_constant_initializer(),
        use_bias=False,
    )(inp)

    # if not const_init:
    deconv = tf.keras.layers.BatchNormalization()(deconv)

    if apply_dropout:
        deconv = tf.keras.layers.Dropout(0.5)(deconv)

    result = tf.keras.layers.ReLU()(deconv)

    return result


def Generator(const_init=True):
    apply_dropout = False

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

    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=get_constant_initializer(),
        activation="tanh",
    )  # (bs, 256, 256, 3)
    u0 = last(u1)

    return tf.keras.Model(inputs=inputs, outputs=d8)


def Discriminator(const_init=True):
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(
        x, 64, 4, apply_batchnorm=False, const_init=const_init
    )  # (bs, 128, 128, 64)
    # (bs, 64, 64, 128)
    down2 = downsample(down1, 128, 4, const_init=const_init)
    # (bs, 32, 32, 256)
    down3 = downsample(down2, 256, 4, const_init=const_init)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=get_constant_initializer(), use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    bn = tf.keras.layers.BatchNormalization()(conv, training=True)

    leaky_relu = tf.keras.layers.LeakyReLU()(bn)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=get_constant_initializer())(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.nn.sigmoid_cross_entropy_with_logits
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.nn.sigmoid_cross_entropy_with_logits
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator = Generator()
discriminator = Discriminator()

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.SGD(1e-4)
discriminator_optimizer = tf.keras.optimizers.SGD(1e-4) 

@tf.function
def train_step(input_image, target):
    # with tf.GradientTape() as disc_tape:
    #     gen_output = generator(input_image, training=True)
    #     disc_real_output = discriminator([input_image, target], training=True)
    #     disc_generated_output = discriminator([input_image, gen_output], training=True)
    #     disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # discriminator_gradients = disc_tape.gradient(disc_loss,
    #                                              discriminator.trainable_variables)
    # discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
    #                                             discriminator.trainable_variables))
    with tf.GradientTape() as gen_tape: 
        gen_output = generator(input_image, training=True)
        # disc_generated_output = discriminator(
        #     [input_image, gen_output], training=True)
        # gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
        #     disc_generated_output, gen_output, target)

    generator_gradients = gen_tape.gradient(gen_output,
                                            generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
                            
    
    return gen_output, None


def generate_results():
    inp = tf.random.normal([BATCH_SIZE, 256, 256, 3])
    tar = tf.random.normal([BATCH_SIZE, 256, 256, 3])

    np.save("input.npy", inp)
    np.save("target.npy", tar)
    for i in range(3):
        g_loss, d_loss = train_step(inp, tar)
        # print("tf produce d_loss:{}, g_loss:{}".format(d_loss.numpy().mean(), g_loss.numpy().mean()))
        print("tf produce g_loss:{}".format(g_loss.numpy().mean()))
    # np.save("d_loss.npy", d_loss.numpy())
    # np.save("g_loss.npy", g_loss.numpy())

if __name__ == '__main__':
    generate_results()
