#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt


def main():
    lr = tfa.optimizers.Triangular2CyclicalLearningRate(
        initial_learning_rate=0.00001,
        maximal_learning_rate=0.0001,
        step_size=10e3 * (8 / 16))
    
    # steps = tf.cast(tf.pow(2, tf.range(0, 20)), tf.float32)
    steps = tf.cast(tf.linspace(0, int(100e3), 1000), tf.float32)
    lr_vals = lr(steps)

    plt.plot(steps.numpy(), lr_vals.numpy())
    plt.show()


if __name__ == '__main__':
    main()
