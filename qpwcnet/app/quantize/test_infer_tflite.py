#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import faulthandler
import cv2

from qpwcnet.core.util import disable_gpu
from qpwcnet.data.augment import image_resize, image_augment
from qpwcnet.data.tfrecord import get_reader
from qpwcnet.core.vis import flow_to_image
from qpwcnet.vis.show import show


def main():
    faulthandler.enable()

    # NOTE(ycho): Mysteriously, tflite segfaults if `channels_first`.
    tf.keras.backend.set_image_data_format('channels_last')

    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    # disable_gpu()

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="/tmp/qpwcnet.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(
        np.random.random_sample(input_shape),
        dtype=np.float32)
    print(input_data.shape)  # 1, 6, 256, 512

    print('set_tensor')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    print('invoke')
    interpreter.invoke()
    print('?')

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[-1]['index'])
    print(output_data.shape)

    def preprocess(ims, flo):
        # 0-255 -> 0.0-1.0
        ims = tf.cast(ims,
                      tf.float32) * tf.constant(1.0 / 255.0,
                                                dtype=tf.float32)
        # resize, no augmentation.
        ims, flo = image_resize(ims, flo, (256, 512))
        # ims, flo = image_augment(ims, flo, (256, 512))
        # 0.0-1.0 -> -0.5, 0.5
        ims = ims - 0.5

        # Convert to correct data format
        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_first':
            ims = einops.rearrange(ims, '... h w c -> ... c h w')
            flo = einops.rearrange(flo, '... h w c -> ... c h w')

        return ims, flo

    if True:
        # TODO(ycho): Cleanup dataset loading pattern for opt-flow datasets.
        glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
        filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
        # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
        # dataset = get_reader(filenames).batch(8).repeat()
        dataset = get_reader(filenames).shuffle(
            buffer_size=32).map(preprocess).batch(1)

        for ims, flo in dataset:
            interpreter.set_tensor(
                input_details[0]['index'],
                ims)  # ims.numpy()?
            interpreter.invoke()
            flo_pred = output_data = interpreter.get_tensor(
                output_details[-1]['index'])
            flo_pred_rgb = flow_to_image(flo_pred)

            show('flo_pred_rgb', flo_pred_rgb[0], True)
            cv2.waitKey(0)

            break


if __name__ == '__main__':
    main()
