# syntax=docker/dockerfile:1

FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    software-properties-common \
    # virtual desktop
    xvfb \
    x11vnc \
    fluxbox \
    # requirements for pyglet (OPENGL)
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    # build requirements for stable retro
    cmake \ 
    build-essential \
    capnproto \ 
    libcapnp-dev \
    libqt5opengl5-dev \ 
    qtbase5-dev \ 
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# ffmpeg
RUN add-apt-repository ppa:ubuntuhandbook1/ffmpeg8 \
    && apt-get update \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && python3.12 -m ensurepip --upgrade \
    && rm -rf /var/lib/apt/lists/*


# Clean up to reduce image size?
RUN apt-get clean 

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# install other packages
COPY --from=project ./requirements.txt ./requirements.txt

RUN python3.12 -m pip install -r requirements.txt

# build stable-retro
RUN git clone https://github.com/farama-foundation/stable-retro.git stable-retro
RUN python3.12 -m pip install -e stable-retro

# build integration tool
WORKDIR /stable-retro
RUN cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY -DPython3_EXECUTABLE=/usr/bin/python3.12 -DPython_INCLUDE_DIR=/usr/include/python3.12
RUN make -j$(grep -c ^processor /proc/cpuinfo)

# install MK2 ROM
COPY --from=project ./rom /usr/src/rom

RUN python3.12 -m retro.import /usr/src/rom


WORKDIR /usr/src/code
# expose needed ports
EXPOSE 8889
EXPOSE 6006
EXPOSE 5900
