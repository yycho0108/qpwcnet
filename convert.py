#!/usr/bin/env python3

from pwcnet import build_network

q_cfg_map = {}
q_cfg_map[Flow] = DelegateConvConfig(['flow'])
q_cfg_map[UpFlow] = DelegateConvConfig(['flow'])
q_cfg_map[Upsample] = DelegateConvConfig(['conv'])
lrelu_cfg = default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
#print('?', q_cfg_map)
#q_cfg_map[tf.keras.activations.swish]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
#q_cfg_map[Mish]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
# q_cfg_map[Mish(
#    mish)]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()


def quantize_annotate_layer(layer):
    L = tf.keras.layers
    # (FeaturesLayer, Flow, UpFlow, Upsample)
    if isinstance(layer, tuple(list(q_cfg_map.keys()))):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, q_cfg_map[layer.__class__])
    elif isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    elif isinstance(layer, tf.keras.layers.LeakyReLU):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, lrelu_cfg)
    else:
        # print(isinstance(layer, tf.keras.layers.LeakyReLU))
        # print(q_cfg_map.keys())
        print('not quantizing class = ', layer.__class__)
    return layer


def quantize_model(model):
    with tfmot.quantization.keras.quantize_scope({
        'FeaturesLayer': FeaturesLayer,
        'Flow': Flow,
        'UpFlow': UpFlow,
        'Upsample': Upsample,
        'Split': Split,
        'DelegateConvConfig': DelegateConvConfig,
        # 'swish': tf.keras.activations.swish,
        # 'mish': Mish(mish),
    }):
        if False:
            qfun = tfmot.quantization.keras.quantize_model
            qmodel = qfun(model)
        else:
            # annotated model
            amodel = tf.keras.models.clone_model(
                model,
                clone_function=quantize_annotate_layer
            )
            # q-aware model
            qmodel = tfmot.quantization.keras.quantize_apply(amodel)
            print('qmodel!')
        qmodel.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])
    qmodel.summary()
    return qmodel


def to_tflite(model):
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    return cvt.convert()


def main():
    net = build_network()
    net.summary()

    tf.keras.utils.plot_model(
        net,
        to_file="net.png",
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    # single forward pass.
    dummy = np.zeros((1, 384, 512, 2), dtype=np.float32)
    out = net(dummy)

    qnet = quantize_model(net)
    net_tfl = to_tflite(qnet)
    # net_tfl = to_tflite(net)
    with open('./pwcnet.tflite', 'wb') as f:
        f.write(net_tfl)


if __name__ == '__main__':
    main()
