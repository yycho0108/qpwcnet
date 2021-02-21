#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import json
import tqdm
from typing import List, Generator, Tuple
from functools import wraps, partial
from collections import OrderedDict
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import einops

from qpwcnet.core.util import file_cache
from qpwcnet.data.augment import photometric_augmentation, restore_shape
from qpwcnet.data.triplet_dataset import TripletDataset
from qpwcnet.data.triplet_dataset_ops import read_triplet_dataset, show_triplet_dataset


def _get_cache_filename(method, self: 'YoutubeVos', *args, **kwargs):
    # print('method = {}'.format(method.__qualname__))
    metadata_file = (Path(self.cache_dir).expanduser() /
                     '{}-{}.json'.format(self.data_type, method.__qualname__))
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
    def _load_metadata(self) -> OrderedDict:
        settings = self.settings
        # Build metadata ... try to preserve order.
        metadata = OrderedDict()
        for d in tqdm.tqdm(sorted(self.dir_.iterdir())):
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


@dataclass
class YoutubeVosTripletSettings:
    dataset: YoutubeVosSettings = YoutubeVosSettings()
    max_gap: int = 0


class YoutubeVosTriplet(TripletDataset):
    """
    Triplet wrapper around YoutubeVos.
    I guess - technically, this is a generic wrapper around `VideoDataset`.
    """

    def __init__(self, cfg: YoutubeVosTripletSettings):
        self.cfg = cfg
        self.dataset = YoutubeVos(cfg.dataset)

    def __iter__(self) -> Tuple[str, str, str]:
        """
        Return a triplet of filenames corresponding to sequential frames.
        """
        for key in self.keys:
            try:
                yield self[key]
            except ValueError as e:
                continue

    def __getitem__(self, key: str) -> Tuple[str, str, str]:
        """ Get a triplet corresponding to a key """
        v = self.dataset.metadata[key]
        n = v['num_frames']

        # Max temporal displacement
        dmax = min((n - 3) // 2, self.cfg.max_gap)
        if dmax < 0:
            raise ValueError(
                'Unable to satisfy max_gap criterion : {} <= {} < 0' .format(
                    dmax, self.cfg.max_gap))

        # displacement = 1 + gap
        d = np.random.randint(1, dmax + 2)

        # Select d-equispaced indices.
        i1 = np.random.randint(d, n - d)
        i0 = i1 - d
        i2 = i1 + d

        # Map to filenames.
        fs = list(self.dataset.get_imgs(key))
        out = (str(fs[i0]), str(fs[i1]), str(fs[i2]))
        return out

    def __len__(self) -> int:
        """ Length of dataset """
        return len(self.dataset)

    @property
    def keys(self) -> List[str]:
        """ RAI keys """
        return self.dataset.get_keys()


def main():
    from qpwcnet.core.util import disable_gpu
    from qpwcnet.data.triplet_dataset_ops import show_triplet_dataset

    disable_gpu()
    dataset = YoutubeVosTriplet(
        YoutubeVosTripletSettings(
            YoutubeVosSettings(
                data_type='train')))
    show_triplet_dataset(dataset)


if __name__ == '__main__':
    main()
