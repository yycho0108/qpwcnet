#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from qpwcnet.core.pwcnet import build_network
from qpwcnet.data.fchairs3d import get_dataset_from_set
from qpwcnet.data.tfrecord import get_reader, read_record
from qpwcnet.data.augment import image_augment, image_resize


def preprocess(ims, flo, data_format='channels_first', base_scale=1.0):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # apply augmentation
    ims, flo = image_augment(ims, flo, (256, 512), base_scale)
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5

    # HWC -> CHW
    if data_format == 'channels_first':
        ims = tf.transpose(ims, (2, 0, 1))
        flo = tf.transpose(flo, (2, 0, 1))

    return ims, flo


def setup_input(batch_size, data_format):

    # Load MPI Sintel dataset.
    # def _preprocess(ims, flo):
    #     return preprocess(ims, flo, data_format, 1.0)
    # glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    # dataset = (tf.data.Dataset.list_files(glob_pattern).interleave(
    #     lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
    #     cycle_length=tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     .shuffle(buffer_size=32)
    #     .map(read_record)
    #     .map(_preprocess)
    #     .batch(batch_size, drop_remainder=True)
    #     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    # Load FlyingChairs3D dataset.
    # FIXME(yycho0108): 0.56 value here is an inevitable result of:
    # - the desire to downsample the image resolution
    # - the requirement that the size of the output image be 256x512.
    def _preprocess_fc3d(ims, flo):
        return preprocess(ims, flo, data_format, 0.56)
    dataset = (get_dataset_from_set()
               .map(_preprocess_fc3d)
               .batch(batch_size, drop_remainder=True)
               .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    # dataset = dataset.concatenate(dataset_fc3d)
    # dataset = dataset.take(1).cache()
    return dataset


def main():
    dataset = setup_input(32, 'channels_first')
    count = 0
    for ims, flo in dataset:
        ims = tf.debugging.check_numerics(
            ims, "nan-ims"
        )
        flo = tf.debugging.check_numerics(
            flo, "nan-flo"
        )
        count += 32
        print(count)


if __name__ == '__main__':
    main()
