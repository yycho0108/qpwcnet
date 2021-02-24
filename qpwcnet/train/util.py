#!/usr/bin/env python3

import tensorflow as tf
from qpwcnet.core.vis import flow_to_image
from pathlib import Path


def convert_savedmodel_to_hdf5(in_file: str, out_file: str):
    """ tf.**.SavedModel -> weights-only hdf5 file """
    import tensorflow as tf
    import tensorflow_addons as tfa
    from qpwcnet.train.loss import AutoResizeMseLoss

    tfa.register_all()
    model = tf.keras.models.load_model(
        in_file,
        custom_objects={
            'AutoResizeMseLoss': AutoResizeMseLoss
        })
    model.save_weights(out_file)


def load_weights(model: tf.keras.Model, model_file: str):
    import tempfile
    import os

    model_file = Path(model_file)

    # If needed, convert to hdf5.
    # (only hdf5 supports by-name loading)
    is_hdf5 = model_file.is_file() and (
        model_file.suffix in ['.hdf5', '.h5'])
    tmp_name = None
    if not is_hdf5:
        fd, tmp_name = tempfile.mkstemp(
            suffix='.hdf5', dir='/tmp/', text=False)
        os.close(fd)

        ctx = mp.get_context('spawn')
        p = ctx.Process(target=convert_savedmodel_to_hdf5,
                        args=(str(model_file), tmp_name))
        p.start()
        p.join()

        model_file = Path(tmp_name)

    # NOTE(ycho): Model arch may be different from original model,
    # thus we load the weights by name.
    # The above conversion to hdf5 is a result of this constraint.
    model.load_weights(str(model_file), by_name=True)

    # Cleanup temporary file, if created.
    if (not is_hdf5) and tmp_name:
        os.remove(tmp_name)


class TensorBoardFlowImage(tf.keras.callbacks.Callback):
    def __init__(self, data, data_format, log_dir: str, tag: str):
        super().__init__()
        self.data_format = data_format
        self.data = data  # dataset? not dataset?
        self.writer = tf.summary.create_file_writer(
            '{}/{}'.format(log_dir, 'flow'))
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        flows = self.model(self.data, training=False)
        flow_images = flow_to_image(flows, data_format=self.data_format)

        with self.writer.as_default():
            tf.summary.image('flow', img, step=epoch)
        self.writer.add_summary(summary, epoch)
