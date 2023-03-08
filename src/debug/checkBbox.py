#!/usr/bin/env python
import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# ROS:
import rospy
from std_msgs.msg import String
from yolov7_msgs.msg import BoundingBoxes
from yolov7_msgs.msg import ObjectCount
from yolov7_msgs.msg import BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_handler(object):
    def __init__(self, topic):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback, )
        self.bbox_sub  = rospy.Subscriber('/yolov7/bounding_boxes', BoundingBoxes, self.yolo_callback)

    def image_callback(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding= "rgb8")
        except CvBridgeError as e:
            print(e)

    def yolo_callback(self,data):
        self.bbox = data

if __name__ == '__main__':
    rospy.init_node('yolov7', anonymous=True)

    topic = '/camera/image_raw_bgr_opencv'
    fps_cap = 100
    
    rospy.wait_for_message(topic, Image, timeout=60)
    rospy.wait_for_message('/yolov7/bounding_boxes', BoundingBoxes,  timeout=60)

    handler = image_handler(topic)
    time.sleep(1)

    # scale_x = int(1080/640)
    # scale_y = int(1920/640)
    # print("SCALE", scale)

    while not rospy.is_shutdown():
        time_start=time.time()
        image = handler.cv_image
        bboxes = handler.bbox
        # image = cv2.resize(image, [640,480], interpolation=cv2.INTER_AREA)

        # print("OBJECT COUNT: ", len(bboxes.bounding_boxes))
        for object in bboxes.bounding_boxes:
            name = object.Class
            probability = object.probability
            bbox_x_min = object.xmin
            bbox_x_max = object.xmax
            bbox_y_min = object.ymin 
            bbox_y_max = object.ymax

            object_center_x = int(((bbox_x_min)+(bbox_x_max)) / 2)
            object_center_y = int(((bbox_y_min)+(bbox_y_max)) / 2)

            probability_round = round(probability,2)

            cv2.rectangle(image, (bbox_x_min,bbox_y_min), (bbox_x_max,bbox_y_max), (255,0,0), 2)
            cv2.circle(image, (object_center_x,object_center_y), 2, (255,255,255), 2)
            cv2.putText(image, str(name)+' %'+str(probability_round*100), (object_center_x+10, object_center_y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # time_end = time.time()
        time.sleep(1/fps_cap)
        time_end = time.time()
        print("FPS: ", 1/(time_end-time_start))
        image = cv2.resize(image, [640,480], interpolation=cv2.INTER_AREA)
        cv2.imshow(topic, image)
        cv2.waitKey(1)






