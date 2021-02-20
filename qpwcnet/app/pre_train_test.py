#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_addons.layers.optical_flow

from qpwcnet.core.pwcnet import build_interpolator
from qpwcnet.data.youtube_vos import YoutubeVos, YoutubeVosSettings, triplet_dataset
from qpwcnet.train.loss import AutoResizeMseLoss
from qpwcnet.app.arg_setup import with_args


def main():
    tf.keras.backend.set_image_data_format('channels_first')
    tfa.register_all()

    # model_file = '/tmp/pwc/run/031/ckpt/058.pb/'
    model_file = '/tmp/pwc/run/037/ckpt/004.pb/'

    # restore
    if False:
        model = build_interpolator(
            input_shape=(256, 512),
            output_multiscale=False)
        # from ckpt
        #print('<restore>')
        #ckpt = tf.train.Checkpoint(model=model)
        #ckpt_mgr = tf.train.CheckpointManager(
        #    ckpt, '/tmp/pwc/run/031/ckpt/', max_to_keep=8)
        #print(ckpt_mgr.latest_checkpoint)

        model.load_weights(model_file)
        # ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
        print('</restore>')
    else:
        # from hdf5
        # model.load_weights(model_file)
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={
                'AutoResizeMseLoss': AutoResizeMseLoss,
            })
    print('Done with model load')

    # Flow only model
    flow_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('up_flow_3').output)

    dataset = YoutubeVos(YoutubeVosSettings(data_type='valid'))
    dataset = triplet_dataset(
        dataset,
        dsize=(256, 512),
        batch_size=1)

    for img0, img1, img2 in dataset:
        # @see pre_train:preprocess()
        img_pair = tf.concat([img0, img2], axis=3)
        img_pair = tf.transpose(img_pair, (0, 3, 1, 2)) - 0.5

        if False:
            flow = flow_model(img_pair)[0].numpy()
            print(flow[0].min())  # tiny
            print(flow[0].max())  # tiny
            print(flow[1].min())  # tiny
            print(flow[1].max())  # tiny
            cv2.imshow('flow-x', flow[0])
            cv2.imshow('flow-y', flow[1])

        if True:
            print(model.inputs[0].shape)  # None, 256, 512, 6
            pred_img1 = model(img_pair)[-1]
            # print(pred_img1.shape)  # list
            pred_img1 = tf.transpose(pred_img1, (0, 2, 3, 1))[0].numpy()
            print(pred_img1.max())
            print(pred_img1.min())
            cv2.imshow('gt', img1[0].numpy())
            cv2.imshow('pred', 0.5 + pred_img1)

        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break
        continue


if __name__ == '__main__':
    main()
