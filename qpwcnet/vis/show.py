#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import einops


def show(key: str, img: np.ndarray, rgb: bool = True,
         data_format: str = None):
    # if applicable, chw -> hwc
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        img = einops.rearrange(img, 'c h w -> h w c')

    # tf->np
    if tf.is_tensor(img) and tf.executing_eagerly():
        img = img.numpy()
    img = np.asarray(img)

    # rgb->bgr for opencv
    if rgb:
        img = img[..., ::-1]

    cv2.namedWindow(key, cv2.WINDOW_NORMAL)
    cv2.imshow(key, img)
