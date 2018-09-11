# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import threading
import time
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from PIL import Image
from StringIO import StringIO

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('num_requests', '1000', 'Number of requests')
tf.app.flags.DEFINE_string('max_concurrent', '1', 'Number of concurrent requests')
tf.app.flags.DEFINE_string('timeout', '10', 'Timeout for each request')
tf.app.flags.DEFINE_string('image_size', '224', 'Image dimensions for Resnet')
FLAGS = tf.app.flags.FLAGS

class Benchmark(object):
    """
    num_requests: Number of requests.
    max_concurrent: Maximum number of concurrent requests.
    """

    def __init__(self, num_requests, max_concurrent):
        self._num_requests = num_requests
        self._max_concurrent = max_concurrent
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            while self._active == self._max_concurrent:
                self._condition.wait()
            self._active += 1

    def wait(self):
        with self._condition:
            while self._done < self._num_requests:
                self._condition.wait()


def _create_rpc_callback(benchmark):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            result = result_future.result().outputs['outputs'].int_val
        benchmark.inc_done()
        benchmark.dec_active()
    return _callback

# def preprocess_image(image_buffer):
#   """Preprocess JPEG encoded bytes to 3D float Tensor."""

#   # Decode the string as an RGB JPEG.
#   # Note that the resulting image contains an unknown height and width
#   # that is set dynamically by decode_jpeg. In other words, the height
#   # and width of image is unknown at compile-time.
#   image = tf.image.decode_jpeg(image_buffer, channels=3)
#   # After this point, all image pixels reside in [0,1)
#   # until the very end, when they're rescaled to (-1, 1).  The various
#   # adjust_* ops all require this range for dtype float.
#   image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#   # Crop the central region of the image with an area containing 87.5% of
#   # the original image.
#   image = tf.image.central_crop(image, central_fraction=0.875)
#   # Resize the image to the original height and width.
#   image = tf.expand_dims(image, 0)
#   image = tf.image.resize_bilinear(
#       image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
#   image = tf.squeeze(image, [0])
#   # Finally, rescale to [-1,1] instead of [0, 1)
#   image = tf.subtract(image, 0.5)
#   image = tf.multiply(image, 2.0)
#   return image

def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  benchmark = Benchmark(int(FLAGS.num_requests), int(FLAGS.max_concurrent))
  # with open(FLAGS.image, 'rb') as f:
  #   # See prediction_service.proto for gRPC request/response details.
  #   data = f.read()
  image = np.array(Image.new("RGB", size=(224,224), color=(256,0,0)))
  # image = image.resize((224, 224), Image.ANTIALIAS)
  height = image.shape[0]
  width = image.shape[1]
  print("Image shape:", image.shape)
  # images = tf.map_fn(preprocess_image, image, dtype=tf.float32)

  start_time = time.time()
  # Send request

  for i in range(int(FLAGS.num_requests)):  
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet50'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, height, width, 3])) # shape=[1], verify_shape=True
    benchmark.throttle()
    result = stub.Predict.future(request, int(FLAGS.timeout))  # 10 secs timeout
    result.add_done_callback(_create_rpc_callback(benchmark))

  benchmark.wait()
  end_time = time.time()

  print()
  print('{} requests ({} max concurrent)'.format(FLAGS.num_requests, FLAGS.max_concurrent))
  print('{} requests/second'.format(int(FLAGS.num_requests)/(end_time-start_time)))
  print('{} Avg time per request'.format((end_time-start_time)/int(FLAGS.num_requests)))

if __name__ == '__main__':
  tf.app.run()
