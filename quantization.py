# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_saved_model('./tf_arcface_mobilefacenet')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()
# with open('mobilefacenet_fp16.tflite', 'wb') as  f:
#   f.write(tflite_quant_model)

import tensorflow as tf
from test import get_idx_list
import cv2
import os
import random
import numpy as np


aligned_faces_dir = './aligned_faces'
_, idx_image_list = get_idx_list(aligned_faces_dir)
random.shuffle(idx_image_list)

converter = tf.lite.TFLiteConverter.from_saved_model('./tf_arcface_mobilefacenet')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
  for i in range(1000):
    idx, img_name = idx_image_list[i]
    origin_img = cv2.imread(os.path.join(aligned_faces_dir, idx, img_name))
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    input = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    yield [input]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

tflite_quant_model = converter.convert()
with open('mobilefacenet_uint8.tflite', 'wb') as  f:
  f.write(tflite_quant_model)