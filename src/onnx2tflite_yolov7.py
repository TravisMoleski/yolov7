# FROM: https://github.com/VikasOjha666/yolov7_to_tflite/blob/main/yoloV7_to_TFlite%20.ipynb

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./content')
tflite_model = converter.convert()


with open('./content/yolov7_model.tflite', 'xb') as f:
  f.write(tflite_model)
     
