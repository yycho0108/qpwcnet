#!/usr/bin/env python3

import tensorflow as tf
from qpwcnet.core.layers import CostVolume, CostVolumeV2


def main():
    search_range = 4
    data_format = 'channels_last'
    cvol_1 = CostVolume(search_range=search_range, data_format=data_format)
    cvol_2 = CostVolumeV2(search_range=search_range, data_format=data_format)

    if data_format == 'channels_first':
        prv = tf.random.normal(shape=(4, 3, 32, 64))
        nxt = tf.random.normal(shape=(4, 3, 32, 64))
        c1 = cvol_1((prv, nxt))
        c2 = cvol_2((prv, nxt))
        print('diff', tf.reduce_sum(c1-c2))
    else:
        prv = tf.random.normal(shape=(4, 32, 64, 3))
        nxt = tf.random.normal(shape=(4, 32, 64, 3))
        c1 = cvol_1((prv, nxt))
        c2 = cvol_2((prv, nxt))
        print('diff', tf.reduce_sum(c1-c2))


if __name__ == '__main__':
    main()
