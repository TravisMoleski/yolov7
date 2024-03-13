docker rm -f yolov7
docker -D run --privileged -it -e  DISPLAY=$DISPLAY \
--privileged \
--runtime=nvidia \
-v /dev/video0:/dev/video0 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /$(pwd)/../src:/yolov7 \
--gpus all \
--network host \
--env DISPLAY=$DISPLAY \
--env QT_X11_NO_MITSHM=1 \
--name=yolov7 \
yolov7:latest