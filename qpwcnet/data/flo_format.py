#!/usr/bin/env python3
import numpy as np


def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise ValueError(
                'Magic number {} incorrect. Invalid .flo file'.format(magic))
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return data.reshape(h, w, 2)
