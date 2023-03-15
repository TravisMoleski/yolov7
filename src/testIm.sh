# python detect_ros.py --weights weights/yolov7.pt --conf 0.55 --img-size 640
python detect.py --weights yolov7-tiny.pt --conf 0.25 --img-size 640 --source ./video/leopard6mm21122022.mp4
# python detect_live.py --weights yolov7.pt --conf 0.25 --img-size 640 --source ./testImg/newyork.jpg