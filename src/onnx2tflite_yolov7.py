# FROM: https://github.com/VikasOjha666/yolov7_to_tflite/blob/main/yoloV7_to_TFlite%20.ipynb

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./weights/onnx/v7tiny')
tflite_model = converter.convert()

with open('./weights/tflite/cocov7-tiny.tflite', 'xb') as f:
  f.write(tflite_model)
     
