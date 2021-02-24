#!/usr/bin/env python3

import tensorflow as tf

from qpwcnet.data.youtube_vos import YoutubeVos, YoutubeVosSettings, triplet_dataset
from qpwcnet.data.youtube_vos import augment_triplet
from qpwcnet.data.augment import photometric_augmentation

import cv2


def main():
    dataset = YoutubeVos(YoutubeVosSettings(data_type='valid'))
    d = triplet_dataset(dataset, dsize=(256, 512), batch_size=8, shuffle=False,
                        prefetch=False, augment=True)
    # d = d.map(augment)

    for img0, img1, img2 in d:
        i0, i1, i2 = img0.numpy(), img1.numpy(), img2.numpy()
        cv2.imshow('i0', i0[0, ..., ::-1])
        cv2.imshow('i1', i1[0, ..., ::-1])
        cv2.imshow('i2', i2[0, ..., ::-1])
        # cv2.imshow('i0a', img0_aug.numpy()[..., ::-1])
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break


if __name__ == '__main__':
    main()
