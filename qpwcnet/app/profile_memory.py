#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf

from qpwcnet.core.pwcnet import build_network
from qpwcnet.app.train import FlowMseLoss
from qpwcnet.core.util import disable_gpu


def main():
    data_format = 'channels_first'
    log_dir = '/tmp/pwc/profile'
    # disable_gpu()

    # No eager
    tf.compat.v1.disable_eager_execution()
    model = build_network(train=True, data_format=data_format)

    # Add loss terms.
    losses = []
    for out in model.outputs:
        losses.append(FlowMseLoss(data_format=data_format))
    model.compile(
        optimizer='adam',
        loss=losses
    )

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
