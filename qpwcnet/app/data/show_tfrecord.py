#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

from qpwcnet.core.util import disable_gpu
from qpwcnet.data.tfrecord import get_reader
from qpwcnet.data.fchairs3d import get_dataset_from_set
from qpwcnet.data.augment import image_augment, image_resize
from qpwcnet.core.warp import tf_warp
from qpwcnet.core.vis import flow_to_image


def normalize(x):
    x = np.asarray(x)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx-mn)


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # return image_resize(ims, flo, (256, 512))
    return image_augment(ims, flo, (256, 512))


def preprocess_fc3d(ims, flo):
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    return image_augment(ims, flo, (256, 512), 0.56)


def compute_stats(size=1024):
    reader = (get_dataset_from_set()
              .map(preprocess_fc3d, num_parallel_calls=tf.data.experimental.AUTOTUNE)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
    count = 0
    means = 0
    for entry in tqdm(reader.as_numpy_iterator(), total=size):
        ims, flo = entry
        mean = np.linalg.norm(flo, axis=-1).mean()
        means += mean
        count += 1
        if count >= size:
            break
    print('mean flow : {}'.format(means/count))


def main():
    disable_gpu()

    # compute_stats()

    if False:
        filename = '/media/ssd/datasets/sintel-processed/sintel.tfrecord'
        reader = get_reader(filename).map(preprocess)
    else:
        reader = get_dataset_from_set().map(preprocess_fc3d)
        # reader = get_dataset().interleave(lambda x: Dataset.from_tensors(x).map(decode_files),
        #                                  cycle_length=tf.data.experimental.AUTOTUNE,
        #                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).map(preprocess)

    reader.shuffle(buffer_size=32)
    for entry in reader.as_numpy_iterator():
        ims, flo = entry
        flo_vis = flow_to_image(flo)
        prv = ims[..., :3]
        nxt = ims[..., 3:]

        #print('prv', prv.min(), prv.max())
        #print('nxt', nxt.min(), nxt.max())
        #print('flo', flo.min(), flo.max())
        #print('flo', np.linalg.norm(flo, axis=-1).mean())

        # show prev reconstructed from nxt.
        # nxt_w = tfa.image.dense_image_warp(nxt[None, ...].astype(
        #    np.float32)/255.0, -flo[None, ..., ::-1]).numpy()
        # nxt_w = tf_warp(nxt[None, ...].astype(
        #    np.float32)/255.0, flo[None, ...]).numpy()

        # flo order : (x,y) == (1,0)
        nxt_w = tfa.image.dense_image_warp(
            nxt[None, ...], -flo[None, ..., ::-1])[0].numpy()
        # nxt_w = tf_warp(nxt[None, ...], flo)[0].numpy()
        print(nxt_w.shape)

        cv2.imshow('prv', prv)
        cv2.imshow('nxt', nxt)
        # cv2.imshow('msk', prv_has_flo.astype(np.float32))
        cv2.imshow('nxt_w', nxt_w)
        cv2.imshow('nxt_w2', nxt_w-prv)

        # bgr, prv=b, nxt=g, r=warp
        overlay = np.stack([
            (prv).mean(axis=-1), (nxt).mean(axis=-1),
            (nxt_w).mean(axis=-1)], axis=-1)
        cv2.imshow('overlay', overlay)
        cv2.imshow('flo', normalize(flo[..., 0]))
        cv2.imshow('flo-vis', flo_vis.numpy())
        k = cv2.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
