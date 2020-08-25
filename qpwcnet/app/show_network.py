#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import tensorflow as tf
from qpwcnet.core.pwcnet import build_network


@tf.function
def step(model, inputs):
    return model(inputs)


def main():
    show_summary = True
    show_keras_plot = True
    show_tensorboard_graph = True

    # Setup logging directory.
    log_dir = '/tmp/pwc/'

    # Build network.
    model = build_network()
    if show_summary:
        model.summary()

    if show_keras_plot:
        tf.keras.utils.plot_model(
            model,
            to_file="../../img/net.png",
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    if show_tensorboard_graph:
        # Prepare summary writer for tensorboard graph visualization.
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = tf.summary.create_file_writer('/tmp/pwc/')

        # Create and run through dummy data.
        dummy_inputs = []
        for input_tensor in model.inputs:
            shape = (1,) + input_tensor.shape[1:]
            dummy_input = np.zeros(shape, input_tensor.dtype.as_numpy_dtype)
            dummy_inputs.append(dummy_input)

        tf.summary.trace_on(graph=True, profiler=True)
        dummy_output = step(model, dummy_inputs)

        # Export trace.
        with writer.as_default():
            tf.summary.trace_export(name='pwcnet_trace',
                                    step=0, profiler_outdir=log_dir)


if __name__ == '__main__':
    main()
