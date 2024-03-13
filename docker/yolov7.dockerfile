FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3.8 python3-pip libgl1-mesa-glx libglib2.0-0 libqt5gui5 wget

RUN pip3 install matplotlib==3.2.2 && \
    pip3 install numpy==1.18.5 && \
    pip3 install opencv-python && \
    pip3 install pillow==7.1.2 && \
    pip3 install pyyaml==5.3.1 && \
    pip3 install requests && \
    pip3 install scipy && \
    pip3 install tqdm && \
    pip3 install torch && \
    pip3 instal torchvision && \
    pip3 install pandas && \
    pip3 install seaborn

RUN mkdir yolov7