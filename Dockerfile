# syntax=docker/dockerfile:1

FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update 
RUN apt-get install -y software-properties-common 
# xvfb (virtual display)
RUN apt-get install -y xvfb
#OpenGL
RUN apt-get install -y libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

# ffmpeg
RUN add-apt-repository ppa:ubuntuhandbook1/ffmpeg8
RUN apt-get update
RUN apt-get install -y ffmpeg

# Install Python 3.12

RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get update 
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev
RUN python3.12 -m ensurepip --upgrade
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

COPY requirements.txt ./requirements.txt

RUN python3.12 -m pip install -r requirements.txt

COPY ./rom /usr/src/rom

RUN python3.12 -m retro.import /usr/src/rom


WORKDIR /usr/src/code

EXPOSE 8889
