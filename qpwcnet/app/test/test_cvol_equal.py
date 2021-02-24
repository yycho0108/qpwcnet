#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from qpwcnet.core.layers import CostVolume, CostVolumeV2


def main():
    for cn in ['channels_first', 'channels_last']:
        tf.keras.backend.set_image_data_format(cn)
        shape = (3, 128, 256) if cn == 'channels_first' else (128, 256, 3)
        prv = tf.keras.Input(shape=shape)
        nxt = tf.keras.Input(shape=shape)
        cvol = CostVolume(4)
        cvol2 = CostVolumeV2(4)

        inputs = (prv, nxt)
        outputs = (cvol((prv, nxt)), cvol2((prv, nxt)))
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile()

        prv_np, nxt_np = (np.random.normal(size=(1,) + shape),
                          np.random.normal(size=(1,) + shape))
        v0, v1 = model.predict((prv_np, nxt_np))
        print((v0 - v1).sum())  # 0.0


if __name__ == '__main__':
    main()
