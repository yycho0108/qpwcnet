#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from qpwcnet.data.tfrecord import get_reader
from qpwcnet.data.augment import image_augment, image_resize
from qpwcnet.core.warp import tf_warp


def normalize(x):
    x = np.asarray(x)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx-mn)


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    return image_resize(ims, flo, (256, 512))
    # return image_augment(ims, flo, (256, 512))


def main():
    filename = '/media/ssd/datasets/sintel-processed/sintel.tfrecord'
    reader = get_reader(filename).map(preprocess)
    reader.shuffle(buffer_size=32)
    for entry in reader.as_numpy_iterator():
        ims, flo = entry
        prv = ims[..., :3]
        nxt = ims[..., 3:]
        print(prv.min(), prv.max())
        print(nxt.min(), nxt.max())

        # show prev reconstructed from nxt.
        # nxt_w = tfa.image.dense_image_warp(nxt[None, ...].astype(
        #    np.float32)/255.0, -flo[None, ..., ::-1]).numpy()
        # nxt_w = tf_warp(nxt[None, ...].astype(
        #    np.float32)/255.0, flo[None, ...]).numpy()
        
        # flo order : (x,y) == (1,0)
        # nxt_w = tfa.image.dense_image_warp(nxt[None, ...].astype(
        #     np.float32), -flo[None, ..., ::-1]).numpy()
        nxt_w = tf_warp(nxt[None,...], flo)[0].numpy()
        print(nxt_w.shape)

        cv2.imshow('prv', prv)
        cv2.imshow('nxt', nxt)
        # cv2.imshow('msk', prv_has_flo.astype(np.float32))
        cv2.imshow('nxt_w', nxt_w)

        # bgr, prv=b, nxt=g, r=warp
        overlay = np.stack([
            (prv).mean(axis=-1), 0*(nxt).mean(axis=-1),
            (nxt_w).mean(axis=-1)], axis=-1)
        cv2.imshow('overlay', overlay)
        cv2.imshow('flo', normalize(flo[..., 0]))
        k = cv2.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
