# This code is to build one client to request the serve response

import grpc
import argparse
import requests
import base64
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras.preprocessing import image


# this part is designed to get and preprocess the image
img_path = '../venv/test_img/test_dog.jpeg'

# this part is designed to set the host configuration
tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')

FLAGS = tf.compat.v1.app.flags.FLAGS
img_size = 224
img_depth = 3


def request_main():
    img = image.load_img(path=img_path, target_size=(img_size, img_size, img_depth))

    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'VGG16'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(img, dtype=tf.float32,
                                                            shape=[1, img_size, img_size, img_depth]))
    result = stub.Predict(request, 200)
    print(result)


if __name__ == "__main__":
    request_main()

