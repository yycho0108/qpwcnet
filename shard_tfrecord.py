#!/usr/bin/env python3

from tqdm import tqdm
from pathlib import Path
import tensorflow as tf


def main():
    in_file = '/media/ssd/datasets/sintel-processed/sintel.tfrecord'
    out_dir = '/media/ssd/datasets/sintel-processed/shards'
    out_pattern = 'sintel-{:04d}.tfrecord'
    num_shards = 32

    raw_dataset = tf.data.TFRecordDataset(in_file, compression_type='ZLIB')
    out_path = Path(out_dir)
    for shard_index in tqdm(range(num_shards)):
        writer = tf.data.experimental.TFRecordWriter(
            str(out_path / out_pattern.format(shard_index)),
            compression_type='ZLIB'
        )
        writer.write(raw_dataset.shard(num_shards, shard_index))


if __name__ == '__main__':
    main()
