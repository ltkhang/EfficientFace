import tensorflow as tf
import cv2
import numpy as np
from test import get_model, get_idx_list, get_feature, cal_dist
import mxnet as mx
import os
import time
from sklearn.preprocessing import normalize

print(tf.__version__)

imported = tf.saved_model.load('./tf_arcface_mobilefacenet/')
concrete_func = imported.signatures['serving_default']
print(concrete_func.structured_outputs)
print(concrete_func.inputs)
img = cv2.imread('./aligned_faces/000/000_0.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data = np.asarray(img, np.float32)
output = concrete_func(tf.convert_to_tensor(np.expand_dims(input_data, axis=0)))
print(output)

arcface_mx = get_model('./model/model-y1-test2', mx.cpu())

print('get list file')
aligned_faces_dir = '.\\aligned_faces'
idx_list, image_idx_list = get_idx_list(aligned_faces_dir)
print(idx_list[0], image_idx_list[0])
features = []
total_time = 0
total_time_arcface = 0
features_arcface = []
for image_idx in image_idx_list:
    idx, img_name = image_idx
    origin_img = cv2.imread(os.path.join(aligned_faces_dir, idx, img_name))
    t = time.time()
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    f = concrete_func(tf.convert_to_tensor(np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)))
    f1 = normalize(f['output0'].numpy()).flatten()
    # print(f1)
    dt = time.time() - t
    total_time += dt
    features.append(f1)
    print('F1', dt)
    #
    t = time.time()
    f2 = get_feature(arcface_mx, origin_img)
    # print(f2)
    # features_arcface.append(f2)
    dt = time.time() - t
    total_time_arcface += dt
    print('F2', dt)
    print('Sim', cal_dist(f1, f2))
print('Avg time',  float(total_time) / len(image_idx_list))
print('Avg arcface', float(total_time_arcface) / len(image_idx_list))

