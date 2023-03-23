# python detect_ros.py --weights weights/yolov7.pt --conf 0.55 --img-size 640
# python detect.py --weights yolov7-tiny.pt --conf 0.25 --img-size 640 --source ./video/leopard6mm21122022.mp4
python3 detect_live.py --weights ./weights/deer-tiny.pt --conf 0.35 --img-size 1120 --source ./video/deer.mp4