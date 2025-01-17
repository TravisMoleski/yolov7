# python detect_ros.py --weights weights/yolov7.pt --conf 0.55 --img-size 640
python3 detect.py --weights ./weights/yolov7.pt --conf 0.25 --img-size 1120 --source ./videos/stopsign.mp4
# python3 detect_live.py --weights ./weights/deer-tiny.pt --conf 0.35 --img-size 1120 --source ./video/deer.mp4
# python3 detect_ros.py --weights ./weights/logos.pt --conf 0.35 --img-size 1120 --source ./video/logo.mp4
# python3 detect_ros.py --weights ./weights/logos.pt --conf 0.30 --img-size 1120 --source ./video/logo.mp4