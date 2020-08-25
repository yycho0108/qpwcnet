#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

from qpwcnet.core.pwcnet import build_network
from qpwcnet.core.warp import tf_warp


def normalize(x):
    x = np.asarray(x)
    minx, maxx = x.min(), x.max()
    return (x - minx) / (maxx-minx)


def main():
    # model_file = './weights/pwc.076.hdf5'
    # model_file = '/tmp/pwc.076.hdf5'
    # model_file = '/tmp/pwc/run/012/ckpt/'
    model_file = '/tmp/pwc/run/013/model.h5'
    model = build_network(train=False)

    # restore
    if False:
        ckpt = tf.train.Checkpoint(
            optimizer=tf.keras.optimizers.Adam(), net=model)
        ckpt_mgr = tf.train.CheckpointManager(
            ckpt, '/tmp/pwc/run/012/ckpt/', max_to_keep=8)
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
    else:
        model.load_weights(model_file)#.expect_partial()
    print('done with load')

    # x = np.random.uniform(0, 255, size=(1, 256, 512, 6)).astype(np.uint8)
    lhs = cv2.imread(
        '/media/ssd/datasets/MPI-Sintel-complete/training/final/market_2/frame_0015.png')
    rhs = cv2.imread(
        '/media/ssd/datasets/MPI-Sintel-complete/training/final/market_2/frame_0016.png')
    lhs = cv2.resize(lhs, (512, 256))
    rhs = cv2.resize(rhs, (512, 256))
    x = np.concatenate([lhs, rhs], axis=-1)[None, ...]
    y = 20.0 * model(x/255.0).numpy()
    rhs_w = tf_warp(rhs[None, ...].astype(np.float32)/255.0,
                    y)[0].numpy()

    cv2.imshow('lhs', lhs)
    cv2.imshow('rhs', rhs)
    cv2.imshow('overlay', rhs//2 + lhs//2)
    cv2.imshow('flow-x', normalize(y[0, ..., 0]))
    cv2.imshow('flow-y', normalize(y[0, ..., 1]))
    cv2.imshow('rhs-w', rhs_w)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
