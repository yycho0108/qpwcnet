#!/usr/bin/env python3

import faulthandler
faulthandler.enable()

import numpy as np
import tensorflow as tf

#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./pwcnet.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

print('set_tensor')
interpreter.set_tensor(input_details[0]['index'], input_data)
print('invoke')
interpreter.invoke()
print('?')

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
