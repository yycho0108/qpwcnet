#!/usr/bin/env python3

import tensorflow as tf
from qpwcnet.core.pwcnet import build_network


@tf.function
def dummy_trace(model, x):
    return model(x)


def main():
    log_dir = '/tmp/pwc-logs/profile'

    model = build_network(train=True)
    writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=True)

    x = tf.zeros(shape=(8, 256, 512, 6), dtype=tf.float32)
    dummy_trace(model, x)
    with writer.as_default():
        tf.summary.trace_export(
            name='model_trace', step=0, profiler_outdir=log_dir)

if __name__ == '__main__':
    main()
