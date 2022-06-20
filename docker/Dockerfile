#FROM pytorch/pytorch:latest
FROM tensorflow/tensorflow:2.4.0-gpu

WORKDIR /root

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY tmp_req.txt .
RUN pip install -r tmp_req.txt
