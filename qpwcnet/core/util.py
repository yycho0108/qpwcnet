#!/usr/bin/env python3

import os
import sys
import tensorflow as tf


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
