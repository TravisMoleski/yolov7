FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3.8 python3-pip libgl1-mesa-glx libglib2.0-0 libqt5gui5

RUN mkdir yolov7