#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2

from qpwcnet.core.util import disable_gpu
from qpwcnet.core.vis import flow_to_image
from qpwcnet.vis.show import show
from qpwcnet.app.train import setup_input


def main():
    disable_gpu()
    data_format = 'channels_first'
    tf.keras.backend.set_image_data_format(data_format)
    dataset = setup_input(8, data_format)

    for imgs, flows in dataset:
        idx = np.random.randint(8)

        prv = imgs[idx, :3]
        nxt = imgs[idx, 3:]
        flo = flows[idx]
        flo_rgb = flow_to_image(flo, data_format=data_format)

        show('prv', 0.5 + prv, True, data_format)
        show('nxt', 0.5 + nxt, True, data_format)
        # FLO corresponds to stuff in `prv`
        show('flo', flo_rgb, True, data_format)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break


if __name__ == '__main__':
    main()
