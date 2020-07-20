#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def normalize(x):
    x = np.asarray(x)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx-mn)


def get_options():
    return tf.io.TFRecordOptions(compression_type='ZLIB')


def get_writer(out_path='/tmp/sintel.tfrecord'):
    tf_record_options = get_options()
    return tf.io.TFRecordWriter(out_path, options=tf_record_options)


def get_reader(filename='/tmp/sintel.tfrecord'):
    return tf.data.TFRecordDataset(
        [filename], compression_type='ZLIB').map(read_record)


def write_record(writer, prv: bytes, nxt: bytes, flo: np.ndarray):
    h, w = flo.shape[:2]
    writer.write(
        tf.train.Example(
            features=tf.train.Features(feature={
                'width': _int64_feature(w),
                'height': _int64_feature(h),
                'prv': _bytes_feature(prv),  # png
                'nxt': _bytes_feature(nxt),  # png
                # tensor
                'flo': _bytes_feature(tf.io.serialize_tensor(flo).numpy())
            })
        ).SerializeToString()
    )


def read_record(serialized_example):
    feature_description = {
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),

        'prv': tf.io.FixedLenFeature((), tf.string),
        'nxt': tf.io.FixedLenFeature((), tf.string),
        'flo': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_example, feature_description)

    prv = tf.io.decode_png(example['prv'])
    nxt = tf.io.decode_png(example['nxt'])
    flo = tf.io.parse_tensor(example['flo'], out_type=tf.float32)
    flo = tf.reshape(flo, (example['height'], example['width'], 2))

    return (prv, nxt, flo)


def test_flo():
    sintel_root = Path('/media/ssd/datasets/MPI-Sintel-complete/')
    flo_file = 'training/flow/ambush_2/frame_0001.flo'
    flo = read_flo(sintel_root / flo_file)

    cv2.imshow('flo', normalize(flo[..., 0]))
    cv2.waitKey(0)


def main():
    test_flo()


if __name__ == '__main__':
    main()
