import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers

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


# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def cross_entropy(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_optimizer = tf.keras.optimizers.SGD(1e-4)
discriminator_optimizer = tf.keras.optimizers.SGD(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(z, images):
    with tf.GradientTape() as gen_tape: 
      generated_images = generator(z, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    with tf.GradientTape() as disc_tape:
      generated_images = generator(z, training=True)
      real_output = discriminator(images, training=True)
      disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def generate_results():
    z = tf.random.normal([BATCH_SIZE, noise_dim])
    img = tf.random.normal([BATCH_SIZE, 28, 28, 1])
    np.save("z.npy", z)
    np.save("img.npy", img)
    g_loss, d_loss = train_step(z, img)
    # print("tf produce d_loss:{}, g_loss:{}".format(d_loss, g_loss))
    np.save("d_loss.npy", d_loss.numpy())
    np.save("g_loss.npy", g_loss.numpy())

if __name__ == '__main__':
    generate_results()