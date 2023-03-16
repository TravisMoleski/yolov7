# FROM: https://github.com/VikasOjha666/yolov7_to_tflite/blob/main/yoloV7_to_TFlite%20.ipynb

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('/catkin_ws/src/yolov7/src/weights/tf')
tflite_model = converter.convert()

with open('/catkin_ws/src/yolov7/src/weights/tf_lite/cocov7-tiny.tflite', 'xb') as f:
  f.write(tflite_model)
     
