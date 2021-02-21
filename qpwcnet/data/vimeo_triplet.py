#!/usr/bin/env python3

from typing import Tuple, Generator, List
from pathlib import Path
from dataclasses import dataclass
from functools import partial
import tensorflow as tf
import cv2

from qpwcnet.data.triplet_dataset import TripletDataset
from qpwcnet.data.triplet_dataset_ops import read_triplet_dataset
from qpwcnet.data.augment import photometric_augmentation, restore_shape


@dataclass
class VimeoTripletSettings:
    root_dir: str = '/media/ssd/datasets/vimeo_triplet/'
    data_type: str = 'train'  # one of (train, test)
    # cache_dir: str = '~/.cache/qpwcnet/'


class VimeoTriplet(TripletDataset):
    """
    Vimeo-Triplet dataset.
    https://toflow.csail.mit.edu
    """

    def __init__(self, cfg: VimeoTripletSettings):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.seq_dir = self.root / 'sequences'
        seq_file = self.root / 'tri_{}list.txt'.format(cfg.data_type)
        if not seq_file.exists():
            raise FileNotFoundError('DNE: {}'.format(seq_file))
        with open(str(seq_file), 'r') as f:
            self.keys_ = sorted(f.read().splitlines())

    def __iter__(self) -> Tuple[str, str, str]:
        """
        Return a triplet of filenames corresponding to sequential frames.
        """
        for key in self.keys_:
            yield self[key]

    def __getitem__(self, key: str) -> Tuple[str, str, str]:
        """ Get a triplet corresponding to a key """
        vid_dir = self.seq_dir / key
        return (str(vid_dir / 'im1.png'),
                str(vid_dir / 'im2.png'), str(vid_dir / 'im3.png'))

    def __len__(self) -> int:
        """ Length of dataset """
        return len(self.keys_)

    @property
    def keys(self) -> List[str]:
        """ RAI keys """
        return self.keys_


def main():
    from qpwcnet.core.util import disable_gpu
    from qpwcnet.data.triplet_dataset_ops import show_triplet_dataset

    disable_gpu()
    dataset = VimeoTriplet(VimeoTripletSettings(data_type='train'))
    show_triplet_dataset(dataset)


if __name__ == '__main__':
    main()
