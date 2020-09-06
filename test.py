import cv2
from glob import glob
import os
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import normalize
import mxnet as mx


def get_model(model_path, ctx, image_size=112):
    layer = 'fc1'
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path + '/model', 0)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size, image_size))])
    model.set_params(arg_params, aux_params)
    return model

def get_feature(model, nimg):
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = normalize(embedding).flatten()
    return embedding

def cal_dist(f1, f2):
    return np.sum(np.square(f1-f2))

def get_idx_list(dataset_dir, ext='bmp'):
    idx_list = [o for o in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, o))]
    image_idx_list = []
    for idx in sorted(idx_list):
        for image_path in glob(os.path.join(dataset_dir, idx, '*.' + ext)):
            image_idx_list.append((idx, os.path.basename(image_path)))
    return idx_list, image_idx_list


if __name__ == '__main__':
    tflite_model = './mobilefacenet.tflite'
    # tflite_model = './mobilefacenet_uint8.tflite'
    interpreter = tf.lite.Interpreter(
        model_path=tflite_model, num_threads=1
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    print('warm up')
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    arcface_mx = get_model('./model/model-y1-test2', mx.cpu())

    print('get list file')
    aligned_faces_dir = '.\\aligned_faces'
    idx_list, image_idx_list = get_idx_list(aligned_faces_dir)
    print(idx_list[0], image_idx_list[0])
    features = []
    total_time = 0
    total_time_arcface = 0
    features_arcface = []
    print(output_details[0]['quantization'])
    for image_idx in image_idx_list:
        idx, img_name = image_idx
        origin_img = cv2.imread(os.path.join(aligned_faces_dir, idx, img_name))
        t = time.time()
        img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        # img = np.array(img, dtype=np.uint8)
        interpreter.set_tensor(input_details[0]['index'], [img])

        interpreter.invoke()
        f = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        if output_details[0]['dtype'] == np.uint8:
            print('scale')
            scale, zero_point = output_details[0]['quantization']
            f = scale * (f - zero_point)
        f1 = normalize(np.expand_dims(np.asarray(f), axis=0)).flatten()
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





