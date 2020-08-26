#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf

from qpwcnet.core.pwcnet import build_network
from qpwcnet.app.train import FlowMseLoss


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

# class MySession(tf.compat.v1.Session):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#    def run(self, *args, **kwargs):
#        super().run(*args, **kwargs,
#                options=run_options, run_metadata=run_metadata)


def main():
    log_dir = '/tmp/pwc/profile'
    disable_gpu()

    # No eager
    tf.compat.v1.disable_eager_execution()
    model = build_network(train=True)
    # model.compile(
    #    optimizer='adam',
    #    loss={t: FlowMseLoss() for t in model.outputs}
    # )
    # tf.keras.backend.clear_session

    run_options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    with tf.compat.v1.Session(graph=tf.compat.v1.keras.backend.get_session().graph, config=config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        sess.run(tf.compat.v1.global_variables_initializer())
        writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)

        # Create dummy input.
        feed_dict = {}
        for input_tensor in model.inputs:
            shape = (1,) + input_tensor.shape[1:]
            dummy_input = np.zeros(shape, input_tensor.dtype.as_numpy_dtype)
            feed_dict[input_tensor] = dummy_input

        _ = sess.run([model.outputs], feed_dict=feed_dict,
                     options=run_options, run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'profile', 0)


if __name__ == '__main__':
    main()
