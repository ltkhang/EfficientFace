# Efficient Lite face

Follow https://github.com/pb-julian/liteface to convert arcface mxnet to tensorflow and tensorflow lite

Error: follow these instruction:

https://discuss.pytorch.org/t/mxnet-to-pytorch-via-mmdnn/88368

change the line 408 in mxnet_parser.py to:

```
weight = self.weight_data.get("fc1_weight").asnumpy()
```

or 

```
weight = self.weight_data.get("fc1_weight").asnumpy().transpose((1, 0))
```

aligned_faces directory from CASIA_FACE_V5

https://github.com/ltkhang/CASIA_FACE_V5_verification

Requirement:

for converting: tensorflow 1

for inference: tensorflow 2, mxnet

test.py: compare tflite (cpu) with mxnet (cpu)

load_tf.py: try to inference on tensorflow gpu

Meaning of 'Sim': ensure the vector from converted model vs mxnet model is not way different