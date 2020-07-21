#!/usr/bin/env python3

import cv2
import numpy as np
from lib.pwcnet import build_network
from lib.warp import tf_warp


def normalize(x):
    x = np.asarray(x)
    minx, maxx = x.min(), x.max()
    return (x - minx) / (maxx-minx)


def main():
    model_file = '/tmp/pwc.076.hdf5'
    net = build_network(train=False)
    net.load_weights(model_file)

    # x = np.random.uniform(0, 255, size=(1, 256, 512, 6)).astype(np.uint8)
    lhs = cv2.imread(
        '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_1/frame_0020.png')
    rhs = cv2.imread(
        '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_1/frame_0021.png')
    lhs = cv2.resize(lhs, (512, 256))
    rhs = cv2.resize(rhs, (512, 256))
    x = np.concatenate([lhs, rhs], axis=-1)[None, ...]
    y = net(x).numpy()
    rhs_w = tf_warp(rhs[None, ...].astype(np.float32)/255.0,
            y)[0].numpy()


    cv2.imshow('lhs', lhs)
    cv2.imshow('rhs', rhs)
    cv2.imshow('flow-x', normalize(y[0, ..., 0]))
    cv2.imshow('flow-y', normalize(y[0, ..., 1]))
    cv2.imshow('rhs-w', rhs_w)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
