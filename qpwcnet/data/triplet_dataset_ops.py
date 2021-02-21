#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple
import tensorflow as tf
from functools import partial

from qpwcnet.data.triplet_dataset import TripletDataset
from qpwcnet.data.augment import photometric_augmentation, restore_shape


def read_and_resize(img: tf.string, dsize: Tuple[int, int]):
    img = tf.io.read_file(img)
    img = tf.io.decode_image(img, expand_animations=False, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, dsize)
    return img


def augment_triplet(a: tf.Tensor, b: tf.Tensor, c: tf.Tensor,
                    dsize: Tuple[int, int],
                    batch_size: int = None, *args, **kwargs):
    x = tf.stack([a, b, c], axis=0)  # 3,{NHWC, NCHW}

    # 1) equal...
    if batch_size is None:
        y = photometric_augmentation(x, z_shape=(1, 1, 1),
                                     *args, **kwargs)
    else:
        # each element of batch must be varied differently.
        y = photometric_augmentation(x, z_shape=(1, batch_size, 1, 1),
                                     *args, **kwargs)

    # Additive gaussian noise
    d0 = () if (batch_size is None) else (batch_size,)
    shape = (1,) + d0 + dsize + (3,)
    y = y + tf.random.normal(shape, 0.0, 0.02)

    # FLIP LR/UD
    y0 = y
    for axis in [-3, -2]:
        if batch_size is not None:
            z = tf.random.uniform(
                [1, batch_size, 1, 1, 1],
                0, 1.0, dtype=tf.float32)
        else:
            z = tf.random.uniform(
                [1, 1, 1, 1],
                0, 1.0, dtype=tf.float32)
        flip = tf.less(z, 0.5)
        y = tf.where(flip, tf.reverse(y, axis=[axis]), y)
    y = restore_shape(y0, y)

    return tf.unstack(y, axis=0)


def read_triplet_dataset(dataset: TripletDataset,
                         dsize: Tuple[int, int],
                         batch_size: int = None,
                         shuffle: bool = True,
                         augment: bool = True,
                         prefetch: bool = True):
    # triplet filenames
    d = tf.data.Dataset.from_generator(
        lambda: (triplet for triplet in dataset),
        (tf.string, tf.string, tf.string))

    # shuffle, etc.
    if shuffle:
        d = d.shuffle(buffer_size=len(dataset))

    # triplet images
    # TODO(ycho): Consider mapping to dataset.img ?
    read_fn = partial(read_and_resize, dsize=dsize)
    d = d.map(lambda a, b, c: (read_fn(a), read_fn(b), read_fn(c)),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # batch
    if batch_size is not None:
        d = d.batch(batch_size, drop_remainder=True)

    if augment:
        aug_fun = partial(augment_triplet, dsize=dsize, batch_size=batch_size)
        d = d.map(aug_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # prefetch
    if prefetch:
        d = d.prefetch(tf.data.experimental.AUTOTUNE)

    return d


def show_triplet_dataset(dataset: TripletDataset):
    import cv2
    from qpwcnet.core.util import disable_gpu
    from qpwcnet.vis.show import show

    dataset = read_triplet_dataset(dataset, dsize=(256, 512))

    _imshow = partial(show, rgb=True, data_format='channels_last')

    for img0, img1, img2 in dataset:
        _imshow('img0', img0)
        _imshow('img1', img1)
        _imshow('img2', img2)
        _imshow('overlay', 0.5 * img0 + 0.5 * img2)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
           break


def main():
    import os
    import cv2
    import numpy as np
    import tempfile

    class DummyTripletDataset(TripletDataset):

        def __iter__(self) -> Tuple[str, str, str]:
            self.file = ''
            fd, self.file = tempfile.mkstemp(
                suffix='.png', dir='/tmp', text=False)
            os.close(fd)

            cv2.imwrite(self.file, np.zeros((256, 256, 3), dtype=np.float32))

            for key in self.keys:
                yield self[key]

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            if self.file:
                os.remove(self.file)
            self.file = ''

        def __getitem__(self, key: str):
            return (self.file, self.file, self.file)

        def __len__(self) -> int:
            return 128

        @property
        def keys(self) -> List[str]:
            return [str(i) for i in range(128)]

    with DummyTripletDataset() as dataset:
        show_triplet_dataset(dataset)


if __name__ == '__main__':
    main()
