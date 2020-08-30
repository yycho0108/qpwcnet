#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

from qpwcnet.core.pwcnet import build_network
from qpwcnet.core.warp import tf_warp
from qpwcnet.data.tfrecord import get_reader
from qpwcnet.data.augment import image_resize, image_augment


def normalize(x):
    x = np.asarray(x)
    minx, maxx = x.min(), x.max()
    return (x - minx) / (maxx-minx)


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # resize, no augmentation.
    # ims, flo = image_resize(ims, flo, (256, 512))
    ims, flo = image_augment(ims, flo, (256, 512))
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5

    # HWC -> CHW
    ims = tf.transpose(ims, (2, 0, 1))
    flo = tf.transpose(flo, (2, 0, 1))

    return ims, flo


def main():
    # model_file = './weights/pwc.076.hdf5'
    # model_file = '/tmp/pwc.076.hdf5'
    # model_file = '/tmp/pwc/run/012/ckpt/'
    # model_file = '/tmp/pwc/run/013/model.h5'
    model_file = '/tmp/pwc/run/006/model.h5'
    # model_file = '/tmp/pwc/run/007/model.h5'
    model = build_network(train=False)

    # restore
    if True:
        # from ckpt
        ckpt = tf.train.Checkpoint(
            optimizer=tf.keras.optimizers.Adam(), net=model)
        ckpt_mgr = tf.train.CheckpointManager(
            ckpt, '/tmp/pwc/run/039/ckpt/', max_to_keep=8)
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
    else:
        # from hdf5
        model.load_weights(model_file)
    print('done with load')

    if False:
        # from image

        # x = np.random.uniform(0, 255, size=(1, 256, 512, 6)).astype(np.uint8)
        lhs = cv2.imread(
            '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_3/frame_0014.png')
        rhs = cv2.imread(
            '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_3/frame_0015.png')
        lhs = cv2.resize(lhs, (512, 256))
        rhs = cv2.resize(rhs, (512, 256))
        x = np.concatenate([lhs, rhs], axis=-1)[None, ...]
        # FIXME(yycho0108): the series of above operations replicate preprocess() data whitening procedure.
        y = model(x/255.0 - 0.5).numpy()
        rhs_w = tf_warp(rhs[None, ...].astype(np.float32)/255.0,
                        y)[0].numpy()

        cv2.imshow('lhs', lhs)
        cv2.imshow('rhs', rhs)
        cv2.imshow('overlay', rhs//2 + lhs//2)
        cv2.imshow('overlay-w', rhs_w/2 + lhs/255.0/2)
        cv2.imshow('flow-x', normalize(y[0, ..., 0]))
        cv2.imshow('flow-y', normalize(y[0, ..., 1]))
        cv2.imshow('rhs-w', rhs_w)
        cv2.waitKey(0)

    if True:
        # from tfrecord
        glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
        filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
        # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
        # dataset = get_reader(filenames).batch(8).repeat()
        dataset = get_reader(filenames).shuffle(
            buffer_size=32).map(preprocess).batch(1)

        for ims, flo in dataset:
            prv = ims[:, :3]
            nxt = ims[:, 3:]
            flo_pred = model(ims)
            # flo_pred = flo
            nxt_w = tf_warp(tf.transpose(nxt, (0, 2, 3, 1)),
                            tf.transpose(flo_pred, (0, 2, 3, 1)))

            # --> numpy
            prv = 0.5 + prv[0].numpy().transpose(1, 2, 0)
            nxt = 0.5 + nxt[0].numpy().transpose(1, 2, 0)
            flo_pred = flo_pred[0].numpy().transpose(1, 2, 0)
            nxt_w = 0.5 + nxt_w[0].numpy()

            cv2.imshow('prv', prv)
            cv2.imshow('nxt', nxt)
            cv2.imshow('overlay', (prv/2 + nxt/2))
            cv2.imshow('overlay-w', (nxt_w/2 + prv/2))

            cv2.imshow('flow-x', normalize(flo_pred[..., 0]))
            cv2.imshow('flow-y', normalize(flo_pred[..., 1]))
            cv2.imshow('flow-x-gt', normalize(flo[0, 0, ...].numpy()))
            cv2.imshow('flow-y-gt', normalize(flo[0, 1, ...].numpy()))

            cv2.imshow('nxt-w', nxt_w)
            k = cv2.waitKey(0)
            if k == 27:
                break


if __name__ == '__main__':
    main()
