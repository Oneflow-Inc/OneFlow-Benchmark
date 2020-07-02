import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

BATCH_SIZE = 32
noise_dim = 100


def const_initializer():
    return tf.constant_initializer(0.002)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            7 * 7 * 256,
            use_bias=False,
            input_shape=(100,),
            kernel_initializer=const_initializer(),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=const_initializer(),
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=const_initializer(),
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=const_initializer(),
            activation="tanh",
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=const_initializer(),
            input_shape=[28, 28, 1],
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            128,
            (5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=const_initializer(),
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_initializer=const_initializer()))

    return model


# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def cross_entropy(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def tf_dcgan_test():
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    z = tf.random.normal([BATCH_SIZE, noise_dim])
    np.save("z.npy", z)
    img = generator(z, training=True)
    logit = discriminator(img, training=True)
    out = generator_loss(logit)
    np.save("out.npy", out.numpy())
