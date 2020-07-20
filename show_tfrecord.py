#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
from tfrecord import get_reader
import tensorflow_addons as tfa


def normalize(x):
    x = np.asarray(x)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx-mn)


def main():
    filename = '/tmp/sintel.tfrecord'
    reader = get_reader(filename)
    reader.shuffle(buffer_size=32)
    for entry in reader.as_numpy_iterator():
        prv, nxt, flo = entry

        # show prev reconstructed from nxt.
        nxt_w = tfa.image.dense_image_warp(nxt[None, ...].astype(
            np.float32)/255.0, -flo[None, ..., ::-1]).numpy()
        #nxt_w = tfa.image.dense_image_warp(nxt[None, ...].astype(
        #    np.float32)/255.0, -flo[None, ..., ::-1]).numpy()

        cv2.imshow('prv', prv)
        cv2.imshow('nxt', nxt)
        # cv2.imshow('msk', prv_has_flo.astype(np.float32))
        cv2.imshow('nxt_w', nxt_w[0])

        # bgr, prv=b, nxt=g, r=warp
        overlay = np.stack([
            (prv/255.0).mean(axis=-1), 0*(nxt/255.0).mean(axis=-1),
            (nxt_w)[0].mean(axis=-1)], axis=-1)
        cv2.imshow('overlay', overlay)
        cv2.imshow('flo', normalize(flo[..., 0]))
        k = cv2.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
