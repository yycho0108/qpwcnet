#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import json
import tqdm
from typing import List, Generator, Tuple
from functools import wraps, partial

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import cv2


def file_cache(name_fn, load_fn=json.load,
               dump_fn=json.dump, binary: bool = True):
    """ Decorator for caching a result from a function to a file. """
    def call_or_load(compute):
        def wrapper(*args, **kwargs):
            filename = name_fn(compute, *args, **kwargs)
            cache_file = Path(filename)
            # Compute if non-existent.
            if not cache_file.exists():
                # Ensure directory exists.
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                result = compute(*args, **kwargs)
                mode = 'wb' if binary else 'w'
                with open(filename, mode) as f:
                    dump_fn(result, f)
                return result

            # O.W. Return from cache.
            mode = 'rb' if binary else 'r'
            with open(filename, mode) as f:
                return load_fn(f)

        return wrapper
    return call_or_load


def _get_cache_filename(method, self: 'YoutubeVos', *args, **kwargs):
    # assert(method == YoutubeVos._load_metadata)
    metadata_file = (Path(self.cache_dir).expanduser() /
                     '{}-metadata.json'.format(self.data_type))
    return metadata_file


@dataclass
class YoutubeVosSettings:
    root_dir: str = '/media/ssd/datasets/youtube_vos/'
    data_type: str = 'valid'
    cache_dir: str = '~/.cache/qpwcnet/'
    img_dir: str = 'JPEGImages'
    img_ext: str = 'jpg'


class YoutubeVos():
    def __init__(self, settings: YoutubeVosSettings):
        self.settings_ = settings
        self.dir_ = (Path(settings.root_dir) /
                     settings.data_type / settings.img_dir)
        self.metadata_ = self._load_metadata()

    @file_cache(_get_cache_filename, binary=False)
    def _load_metadata(self):
        settings = self.settings
        # Build metadata.
        metadata = {}
        for d in tqdm.tqdm(self.dir_.iterdir()):
            num_frames = len(list(d.glob('*.{}'.format(settings.img_ext))))
            metadata[d.name] = {'num_frames': num_frames}

        return metadata

    def __len__(self):
        return len(self.metadata_)

    def get_keys(self) -> List[str]:
        return self.metadata.keys()

    def get_imgs(self, key: str) -> Generator[str, None, None]:
        # assert key in self.metadata
        settings = self.settings
        vid_dir = self.dir_ / key
        for img in sorted(vid_dir.glob('*.{}'.format(settings.img_ext))):
            yield img

    @property
    def settings(self):
        return self.settings_

    @property
    def root_dir(self):
        return self.settings_.root_dir

    @property
    def data_type(self):
        return self.settings_.data_type

    @property
    def cache_dir(self):
        return self.settings_.cache_dir

    @property
    def img_dir(self):
        return self.settings_.img_dir

    @property
    def metadata(self):
        return self.metadata_


def as_generator(dataset: YoutubeVos, max_gap=0):
    for k, v in dataset.metadata.items():
        n = v['num_frames']

        # Max temporal displacement
        dmax = min((n - 3) // 2, max_gap)
        if dmax < 0:
            continue

        # displacement = 1 + gap
        d = np.random.randint(1, dmax + 2)

        # Select d-equispaced indices.
        i1 = np.random.randint(d, n - d)
        i0 = i1 - d
        i2 = i1 + d

        # Map to filenames.
        fs = list(dataset.get_imgs(k))
        out = (str(fs[i0]), str(fs[i1]), str(fs[i2]))
        yield out


def as_tf_dataset(dataset: YoutubeVos, *args, **kwargs) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(
        lambda: as_generator(dataset, *args, **kwargs),
        (tf.string, tf.string, tf.string))


def read_and_resize(img: tf.string, dsize: Tuple[int, int]):
    img = tf.io.read_file(img)
    img = tf.io.decode_image(img, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, dsize)
    # normalize to -0.5~0.5
    img -= 0.5
    return img


def triplet_dataset(dataset: YoutubeVos,
                    dsize: Tuple[int, int] = (256, 256),
                    batch_size: int = None,
                    shuffle: bool = True,
                    prefetch: bool = True
                    ):
    # triplet filenames
    d = as_tf_dataset(dataset, max_gap=0)

    # shuffle, etc.
    if shuffle:
        d = d.shuffle(buffer_size=len(dataset))

    # triplet images
    read_fn = partial(read_and_resize, dsize=dsize)
    d = d.map(lambda a, b, c: (read_fn(a), read_fn(b), read_fn(c)),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # batch
    if batch_size is not None:
        d = d.batch(batch_size)

    # prefetch
    if prefetch:
        d = d.prefetch(tf.data.experimental.AUTOTUNE)

    return d


def main():
    dataset = YoutubeVos(YoutubeVosSettings(data_type='valid'))
    #key = random.choice(list(dataset.get_keys()))
    #imgs = dataset.get_imgs(key)
    #for img in imgs:
    #    print(img)
    #print(len(dataset.metadata))
    d = triplet_dataset(dataset)
    for img0, img1, img2 in d:
        i0, i1, i2 = img0.numpy(), img1.numpy(), img2.numpy()
        cv2.imshow('i0', i0)
        cv2.imshow('i1', i1)
        cv2.imshow('i2', i2)
        cv2.waitKey(0)
        break


if __name__ == '__main__':
    main()
