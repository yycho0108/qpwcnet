#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_addons.layers.optical_flow
import einops
from pathlib import Path
import logging

from qpwcnet.core.pwcnet import build_interpolator
from qpwcnet.core.vis import flow_to_image
from qpwcnet.data.youtube_vos import YoutubeVos, YoutubeVosSettings, triplet_dataset
from qpwcnet.data.vimeo_triplet import VimeoTriplet, VimeoTripletSettings, triplet_dataset as triplet_dataset_v
from qpwcnet.train.loss import AutoResizeMseLoss
from qpwcnet.app.arg_setup import with_args
from qpwcnet.vis.show import show


def _show(key, img, data_format: str = None):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    # Need to convert due to tf.Tensor not supporting assignment.
    if tf.is_tensor(img) and tf.executing_eagerly():
        img = img.numpy()

    # NOTE(ycho): In order to figure out if we have aligned the frames correctly,
    # rather than just copying the inputs, draw a grid on top of the image.
    if data_format == 'channels_first':
        img[..., :, :, ::32] = 1.0
        img[..., :, ::32, :] = 1.0
    else:
        img[..., :, ::32, :] = 1.0
        img[..., ::32, :, :] = 1.0

    return show(key, img, True, data_format)


def main():
    data_format = 'channels_first'
    tf.keras.backend.set_image_data_format(data_format)

    # model_file = '/tmp/pwc/run/031/ckpt/058.pb/'
    # model_file = '/tmp/pwc/run/044/ckpt/020.pb/'
    model_file = '/tmp/pwc/run/054/ckpt/002.ckpt/'
    model_file = Path(model_file)
    multi_output = True
    if False:
        # Build + Restoration
        is_hdf5 = model_file.is_file() and (
            model_file.suffix in ['.hdf5', '.h5'])
        if is_hdf5:
            # NOTE(ycho):
            # This is the only possibility to set output_multiscale=False.
            model = build_interpolator(
                input_shape=(256, 512),
                output_multiscale=False)
            multi_output = False
            model.load_weights(model_file)
        else:
            model = build_interpolator(
                input_shape=(256, 512),
                output_multiscale=True)

            # NOTE(ycho): because tf2 loading scheme is dumb,
            # we cannot load by name with .tf format (not HDF5);
            # this means we need to also set output_multiscale=True.
            model.load_weights(model_file + '/variables/variables',
                               by_name=False)
    else:
        # Load directly from tf.SavedModel.
        tfa.register_all()
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={
                'AutoResizeMseLoss': AutoResizeMseLoss,
            })
    logging.info('Done with model load')

    # Flow-only model...
    flow_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('up_flow_3').output)

    # FIXME(ycho): Ability to select dataset

    #dataset = YoutubeVos(YoutubeVosSettings(data_type='valid'))
    #dataset = triplet_dataset(
    #    dataset,
    #    dsize=(256, 512),
    #    batch_size=1)

    dataset = VimeoTriplet(VimeoTripletSettings(data_type='test'))
    dataset = triplet_dataset_v(dataset, dsize=(256, 512),
                                augment=False,
                                batch_size=1)

    for img0, img1, img2 in dataset:
        img_pair = tf.concat([img0, img2], axis=3)

        # @see pre_train:preprocess()
        if data_format == 'channels_first':
            img_pair = einops.rearrange(img_pair, 'n h w c -> n c h w')
        img_pair -= 0.5

        if True:
            flow = flow_model(img_pair)[0].numpy()
            flow_rgb = flow_to_image(flow, data_format=data_format)
            _show('flow', flow_rgb, data_format)

        if True:
            pred_img1 = model(img_pair)

            # Take the last (full-res) image in case of multi output.
            if multi_output:
                pred_img1 = pred_img1[-1]

            overlay = 0.5 * img0[0] + 0.5 * img2[0]
            _show('prv', img0[0], 'channels_last')
            _show('nxt', img2[0], 'channels_last')
            _show('pred', 0.5 + pred_img1[0], data_format=data_format)
            _show('ground-truth', img1[0], 'channels_last')
            _show('overlay', overlay, 'channels_last')

        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break
        continue


if __name__ == '__main__':
    main()
