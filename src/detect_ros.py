#!/usr/bin/env python

import argparse
import time
from pathlib import Path
import numpy as np
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# ROS:
from std_msgs.msg import String
from yolov7_msgs.msg import BoundingBoxes
from yolov7_msgs.msg import ObjectCount
from yolov7_msgs.msg import BoundingBox

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class image_handler(object):
    def __init__(self, topic):
        self.bridge = CvBridge()

        # print("GETTING", topic)

        self.image_sub = rospy.Subscriber(topic, Image, self.callback, )
        self.image_pub = rospy.Publisher(topic+'_yolo',Image, queue_size=1)

        self.bbox_pub  = rospy.Publisher('/yolov7/bounding_boxes', BoundingBoxes, queue_size=1)

    def callback(self,image_data):
        # print("CALLBACK")
        try:
            # print("OOOOOOOOOOF")
            # self.cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding= "rgb8")
            self.cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
            # self.yolo_img = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2RGB)

            # self.image_pub_opencv.publish(self.bridge.cv2_to_imgmsg(self.yolo_img, "bgr8"))
            # self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))

        except CvBridgeError as e:
            print(e)


def detect(topic):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    save_txt = False
    trace = False
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        # ('rtsp://', 'rtmp://', 'http://', 'https://'))

    webcam = False
    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    # set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
        # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    view_img = False
    save_img = False
    save_txt = False

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    im_hand = image_handler(topic)
    time.sleep(3)


    # for path, img, im0s, vid_cap in dataset:
    while not rospy.is_shutdown():
        # Process detections
        bounding_boxes = BoundingBoxes()
        bounding_boxes.image_header.stamp = rospy.Time.now()

        t_start = time.time()
        img0 = im_hand.cv_image
        # h0, w0 = img0.shape[:2]  # orig hw
        # r = imgsz / max(h0, w0)  # resize image to img_size
        # if r != 1:
        #     img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA )

        # Letterbox
        img = letterbox(img0, imgsz, stride=32)[0]
        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        box_list = [None] * len(pred[0])
        count = 0
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # else:
            #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg?
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print(det)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # print(det)
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # sys.exit(1)
                # # Write results
                for *xyxy, conf, cls in reversed(det):
                    bbox = BoundingBox()
                    # print(float(conf), names[int(cls)])
                    # if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    if view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    bbox.xmin = int(xyxy[0])
                    bbox.xmax = int(xyxy[2])
                    bbox.ymin = int(xyxy[1])
                    bbox.ymax = int(xyxy[3])

                    bbox.probability = float(conf)
                    bbox.Class = f'{names[int(cls)]}'
                    bbox.id = int(cls)
                    box_list[count] = bbox

                    count += 1

                bounding_boxes.bounding_boxes = box_list
                bounding_boxes.header.stamp = rospy.Time.now()
                # rospy.loginfo(bounding_boxes)

                if view_img:
                    resize = cv2.resize(img0, [640, 480],interpolation = cv2.INTER_AREA)
                    cv2.imshow(topic, resize)
                    cv2.waitKey(1)  # 1 millisecond

                t_end = time.time()


                ros_image = Image(encoding="rgb8")
                # Create the header
                ros_image.header.stamp = rospy.Time.now()
                # ros_image.header.frame_id = id
                # Fill the image data 
                ros_image.height, ros_image.width = img0.shape[:2]
                ros_image.data = img0.ravel().tobytes() # or .tostring()
                ros_image.step=ros_image.width
                
                # im_hand.image_pub.publish(im_hand.bridge.cv2_to_imgmsg(img0, "bgr8"))
                im_hand.image_pub.publish(ros_image)
                im_hand.bbox_pub.publish(bounding_boxes)
                print("UPDATE RATE:", 1/(t_end-t_start))

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # # Stream results
            # if view_img:
            #     print("SHOWING...?")
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     #print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.60, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    rospy.init_node('yolov7', anonymous=True)

    topic = '/camera/image_raw_bgr_opencv'
    
    rospy.wait_for_message(topic, Image, timeout=60)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7-tiny.pt']:
                detect(topic)
                strip_optimizer(opt.weights)
        else:
            detect(topic)
 