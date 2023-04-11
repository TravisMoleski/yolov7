
import cv2
import time
import requests
import random
import numpy as np

import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

import sys

# ROS:
from std_msgs.msg import String
from yolov7_msgs.msg import BoundingBoxes
from yolov7_msgs.msg import ObjectCount
from yolov7_msgs.msg import BoundingBox

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

class image_handler(object):
    def __init__(self, topic):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(topic, Image, self.callback, )
        self.image_pub = rospy.Publisher(topic+'_yolo',Image, queue_size=1)
        self.bbox_pub  = rospy.Publisher('/yolov7/bounding_boxes', BoundingBoxes, queue_size=1)

    def callback(self,image_data):
        try:
            self.cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)

        except CvBridgeError as e:
            print(e)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def publish_image(im, im_hand):
    ros_image = Image(encoding="bgr8")
    # Create the header
    ros_image.header.stamp = rospy.Time.now()
    # ros_image.header.frame_id = id
    # Fill the image data 
    ros_image.height, ros_image.width = im.shape[:2]
    ros_image.data = im.ravel().tobytes() # or .tostring()
    ros_image.step=ros_image.width
    
    # im_hand.image_pub.publish(im_hand.bridge.cv2_to_imgmsg(img0, "bgr8"))
    im_hand.image_pub.publish(ros_image)


#Name of the classes according to class indices.
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']


names = ['deer', 'starbucks', 'ou_logo', 'tim_hortons', 'kroger', 'traffic_cone']

#Creating random colors for bounding box visualization.
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

rospy.init_node('yolov7', anonymous=True)
topic = '/usb_cam/image_raw_bgr_opencv'
view_im = True
cuda    = True
w = "./weights/onnx_yv7/logos.onnx"
     

# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
print(ort.get_available_providers())

providers = ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
# sys.exit(1)

rospy.wait_for_message(topic, Image, timeout=60)
im_hand = image_handler(topic)
time.sleep(3)

while not rospy.is_shutdown():

    # img = cv2.imread('./testImg/newyork.jpg')
    img = im_hand.cv_image
    bounding_boxes = BoundingBoxes()
    bounding_boxes.image_header.stamp = rospy.Time.now()

    t_start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}

    # ONNX inference
    outputs = session.run(outname, inp)[0]

    ori_images = [img.copy()]

    #Visualizing bounding box prediction.
    box_list = []
    count = 0
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)

        name = names[cls_id]
        color = colors[name]
        name_score = name + ' '+str(score)

        rect1 = tuple(np.array(box[:2]).astype(int))
        rect2 = tuple(np.array(box[2:]).astype(int))
        # print(rect1)

        bbox = BoundingBox()

        bbox.xmin = int(rect1[0])
        bbox.xmax = int(rect2[0])
        bbox.ymin = int(rect1[1])
        bbox.ymax = int(rect2[1])

        bbox.probability = float(score)

        bbox.Class = f'{name}'
        bbox.id = int(cls_id)

        if float(score) > 0.30:
            box_list.append(bbox)

        # print(bbox)

        count += 1

        if view_im:
            cv2.rectangle(image,rect1,rect2,color=color,thickness=3)
            cv2.putText(image,name_score,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=3)  


    bounding_boxes.bounding_boxes = box_list
    # print(bounding_boxes)

    bounding_boxes.header.stamp = rospy.Time.now()
    im_hand.bbox_pub.publish(bounding_boxes)
        # Image.fromarray(ori_images[0])

    if view_im:
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            resize = cv2.resize(image, [640, 480],interpolation = cv2.INTER_AREA)
            cv2.imshow("YOLOV7 Onnx", resize)
            cv2.waitKey(1)

    # publish_image(image, im_hand)

    t_end = time.time()
    print("YOLO INFERENCE FPS: ", 1 /(t_end-t_start))

 