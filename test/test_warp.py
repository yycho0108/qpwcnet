#!/usr/bin/env python3

import tensorflow as tf
import cv2

from qpwcnet.core.layers import Warp, WarpV2
from qpwcnet.core.util import disable_gpu


def main():
    disable_gpu()

    data_format = 'channels_last'
    warp_1 = Warp(data_format=data_format)
    warp_2 = WarpV2(data_format=data_format)

    if data_format == 'channels_first':
        img = tf.random.uniform(shape=(4, 3, 32, 64))
        flo = tf.random.normal(shape=(4, 2, 32, 64))
        c1 = warp_1((img, flo))
        c2 = warp_2((img, flo))
        print('diff', tf.reduce_mean(c1-c2))
    else:
        img = tf.random.uniform(shape=(4, 32, 64, 3))
        flo = tf.random.normal(shape=(4, 32, 64, 2))
        c1 = warp_1((img, flo))
        c2 = warp_2((img, flo))
        print('diff', tf.reduce_mean(c1-c2))

        cv2.imshow('c1', c1[0].numpy())
        cv2.imshow('c2', c2[0].numpy())
        cv2.imshow('diff', tf.abs(c1-c2)[0].numpy())
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
