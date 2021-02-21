#!/usr/bin/env python3

import os
import sys
import tensorflow as tf
import json

from functools import partial
from collections import OrderedDict
from pathlib import Path


def disable_gpu():
    try:
        # Disable on os level.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Also disable through tf api.
        tf.config.set_visible_devices([], 'GPU')
        # Validate that the devices are NOT on the GPU.
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            print('Device : {}'.format(device))
            assert device.device_type != 'GPU'
    except Exception as e:
        # If there are any exceptions, abort and exit immediately.
        print('Exception while disabling cuda : {}'.format(e))
        sys.exit(0)


json_load = partial(json.load, object_pairs_hook=OrderedDict)


def file_cache(name_fn, load_fn=json_load,
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
