#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np
import cv2
import re

import tensorflow as tf
import tensorflow_io as tfio


def decode_pfm(filename):
    with open(filename.numpy(), 'rb') as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data, scale


def decode_flo(filename):
    data, scale = decode_pfm(filename)
    return data[..., :2]*scale


def decode_files(f_prv, f_nxt, f_flo):
    prv = tfio.image.decode_webp(tf.io.read_file(f_prv))
    nxt = tfio.image.decode_webp(tf.io.read_file(f_nxt))
    flo = tf.py_function(decode_flo, [f_flo], Tout=tf.float32)
    ims = tf.concat([prv[..., :3], nxt[..., :3]], axis=-1)
    return ims, flo


def get_generator(fc3d_root='/media/hdd/datasets/FlyingThings3D/'):
    glob_pattern = 'frames_finalpass_webp/TRAIN/*/*/left/*.webp'
    flow_format = 'optical_flow/TRAIN/{}/{}/into_future/left/OpticalFlowIntoFuture_{}_L.pfm'

    fc3d_root = Path(fc3d_root)
    for f_img in fc3d_root.glob(glob_pattern):
        # Extract hierarchy.
        subset = f_img.parents[2].name
        scene = f_img.parents[1].name

        # Format filenames.
        f_prv = f_img
        f_nxt = f_img.parent / '{:04d}.webp'.format(int(f_img.stem) + 1)
        f_flo = fc3d_root / flow_format.format(subset, scene, f_img.stem)

        # Return if image pair exists.
        if not f_nxt.exists():
            continue
        yield str(f_prv), str(f_nxt), str(f_flo)


def get_dataset(fc3d_root='/media/hdd/datasets/FlyingThings3D/'):
    def _get_generator():
        return get_generator(fc3d_root)
    dataset = tf.data.Dataset.from_generator(
        _get_generator, output_types=(tf.string, tf.string, tf.string))
    return dataset


def _count_lines(filename):
    def _blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
        return (sum(bl.count("\n") for bl in _blocks(f)))


def get_dataset_from_set(set_file: str = '/home/jamiecho/Repos/Ravel/qpwcnet/data/f3d_set.txt'):
    def _files_from_line(line):
        filenames = tf.strings.split(line, sep=' ', maxsplit=3)
        return filenames[0], filenames[1], filenames[2]
    l = _count_lines(set_file)
    return (tf.data.TextLineDataset(set_file).shuffle(l).map(_files_from_line)
            .map(decode_files, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False))


def test():
    FC3D_ROOT = Path('/media/hdd/datasets/FlyingThings3D/')
    for f_img in FC3D_ROOT.glob('frames_finalpass_webp/TRAIN/*/*/left/*.webp'):
        # Extract hierarchy.
        subset = f_img.parents[2].name
        scene = f_img.parents[1].name

        # Format filenames.
        f_prv = f_img
        f_nxt = f_img.parent / '{:04d}.webp'.format(int(f_img.stem) + 1)
        f_flo = FC3D_ROOT / \
            'optical_flow/TRAIN/{}/{}/into_future/left/OpticalFlowIntoFuture_{}_L.pfm'.format(
                subset, scene, f_img.stem)

        print(f_prv)
        print(f_nxt, f_nxt.exists())
        print(f_flo)
        break

    # print(list(FC3D_ROOT.glob('frames_finalpass_webp/TRAIN/*/*/left/*.webp'))[0]) # -> 22390
    # print(len(list(FC3D_ROOT.glob('frames_finalpass_webp/TRAIN/*/*/left/*.webp')))) # -> 22390

    # FC3D_ROOT / 'frames_finalpass_webp' / 'TRAIN' / 'A' / '0000' /  'left' /

    #img_bytes = tf.io.read_file('/media/hdd/datasets/FlyingThings3D/frames_finalpass_webp/TRAIN/A/0000/left/0006.webp')
    #img_rgba = tfio.image.decode_webp(img_bytes)
    #cv2.imshow('img', img_rgba.numpy())
    # cv2.waitKey(0)


def main():
    from tqdm import tqdm
    gen = get_generator()
    with open('/tmp/f3d_set.txt', 'w') as fout:
        for p, n, f in tqdm(gen):
            fout.write('{} {} {}\n'.format(p, n, f))

    # dataset = tf.data.Dataset.from_generator(
    #    get_generator, output_types=(tf.string, tf.string, tf.string))
    #dataset = dataset.map(decode_files)
    # for ims, flo in dataset:
    #    print(ims.shape)
    #    print(flo.shape)
    #    break


if __name__ == '__main__':
    main()
