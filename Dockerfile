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

## Install requirements for the StableRetro integration UI
RUN apt-get install -y cmake build-essential
RUN apt-get install -y capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev zlib1g-dev
RUN apt-get install -y git

# Clean up to reduce image size?
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

COPY requirements.txt ./requirements.txt

RUN python3.12 -m pip install -r requirements.txt

RUN git clone https://github.com/farama-foundation/stable-retro.git stable-retro
RUN python3.12 -m pip install -e stable-retro

COPY ./rom /usr/src/rom

RUN python3.12 -m retro.import /usr/src/rom

RUN apt-get update && apt-get install -y \
    x11vnc \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src/code

EXPOSE 8889
EXPOSE 6006
EXPOSE 5900
