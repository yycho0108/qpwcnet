#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from qpwcnet.core.layers import WarpV2
from qpwcnet.vis.show import show
import cv2


def main():
    """
    Optical flow specification can be somewhat confusing.
    The options are:
    flow => {prv|nxt}[i,j] == {nxt|prv}[i {+|-} flow[i,j,{0,1}], j {+|-} flow[i,j,{1,0}]

    i.e. the following:

    * Whether the flow is specified with inverted signs
    * Whether the flow coordinates correspond to prv/nxt image
    * Whether the flow order is transposed i.e. (i,j)'==(x,y)

    Which total to 8 possibilities. The below script basically
    permutes through all options and tries to figure out which convention it uses.
    """
    tf.keras.backend.set_image_data_format('channels_last')
    warp = WarpV2()

    nxt = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # 3x3 image
    flo = [0, 1]  # +1 in j-dir

    nxt = np.float32(nxt).reshape(1, 3, 3, 1)
    flo = np.float32([1, 0]).reshape(1, 1, 1, 2)
    prv = warp((nxt, flo))

    show('prv', prv[0], False)
    show('nxt', nxt[0], False)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
