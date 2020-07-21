#!/usr/bin/env python3

from tqdm import tqdm
from pathlib import Path
import numpy as np
import tensorflow as tf

from lib.flo_format import read_flo
from lib.tfrecord import write_record, get_writer


def main():
    sintel_root = Path('/media/ssd/datasets/MPI-Sintel-complete/')
    img_root = sintel_root / 'training/final/'
    flo_root = sintel_root / 'training/flow/'
    out_path = '/tmp/sintel.tfrecord'

    with get_writer(out_path) as w:
        flos = list(flo_root.glob('*/*.flo'))
        for flo in tqdm(flos):
            # metadata
            seq = flo.parent.name
            index = int(flo.stem.split('_')[-1])

            # get targets
            prv_name = 'frame_{:04d}.png'.format(index)
            nxt_name = 'frame_{:04d}.png'.format(index+1)

            # format to file
            prv = (img_root / flo.parent.name / prv_name)
            nxt = (img_root / flo.parent.name / nxt_name)

            with open(prv, 'rb') as f:
                prv_bytes = f.read()
            with open(nxt, 'rb') as f:
                nxt_bytes = f.read()
            flo_array = read_flo(flo)

            write_record(w, prv_bytes, nxt_bytes, flo_array)


if __name__ == '__main__':
    main()
