FROM nvidia/cuda:12.0.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 


# RUN sed -i 's@/archive.ubuntu.com/@/ftp.sjtu.edu.cn/@g' /etc/apt/sources.list
# RUN sed -i 's@/security.ubuntu.com/@/ftp.sjtu.edu.cn/@g' /etc/apt/sources.list
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.aliyun.com/@g' /etc/apt/sources.list
RUN sed -i 's@/security.ubuntu.com/@/mirrors.aliyun.com/@g' /etc/apt/sources.list

RUN apt update -y && \
    apt install -y ca-certificates &&\
    apt upgrade -y
RUN apt clean

# RUN apt update -y && \
#     apt upgrade -y

RUN apt install -y \
    build-essential \
    git \
    libeigen3-dev \
    libpcap-dev \
    python3 \
    python3-dev \
    python3-pip \
    libpcl-dev \
    cmake \
    libopencv-dev \
    python3-opencv \
    libgoogle-glog-dev \
    protobuf-compiler \
    libprotoc-dev \
    libzmq3-dev \
    nlohmann-json3-dev \
    libwebsockets-dev \
    wget \
    unzip
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install -U pip

WORKDIR /
RUN git clone https://github.com/zaphoyd/websocketpp.git
WORKDIR /websocketpp
RUN cmake . -DCMAKE_BUILD_TYPE=Release
RUN make -j$(nproc)
RUN make install
WORKDIR /
RUN rm -rf /websocketpp


WORKDIR /
RUN wget https://codeload.github.com/foxglove/ws-protocol/tar.gz/refs/tags/releases/cpp/v1.2.0 -O ws-protocol.tar.gz
RUN tar -xvf /ws-protocol.tar.gz
WORKDIR /ws-protocol-releases-cpp-v1.2.0/cpp/foxglove-websocket
RUN cmake . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
RUN make -j$(nproc)
RUN make install
WORKDIR /
RUN rm -rf /ws-protocol-releases-cpp-v1.2.0
RUN rm -f /ws-protocol.tar.gz

WORKDIR /
RUN wget https://github.com/FFmpeg/nv-codec-headers/releases/download/n12.2.72.0/nv-codec-headers-12.2.72.0.tar.gz
RUN tar -xvf /nv-codec-headers-12.2.72.0.tar.gz
RUN cd /nv-codec-headers-12.2.72.0 && make install
WORKDIR /
RUN rm -rf /nv-codec-headers-12.2.72.0
RUN rm -f /nv-codec-headers-12.2.72.0.tar.gz

RUN apt install -y \
    yasm \
    libtool \
    libc6 \
    libc6-dev \
    libnuma1 \
    libnuma-dev

WORKDIR /
RUN wget https://github.com/FFmpeg/FFmpeg/archive/refs/heads/release/7.0.zip -O ffmpeg.zip
RUN unzip ffmpeg.zip
WORKDIR /FFmpeg-release-7.0
RUN ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
RUN make -j$(nproc)
RUN make install
WORKDIR /
RUN rm -rf /FFmpeg-release-7.0
RUN rm -f /ffmpeg.zip

WORKDIR /
RUN wget https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
RUN mkdir /nvcomp
RUN mv /nvcomp_3.0.6_x86_64_12.x.tgz /nvcomp
WORKDIR /nvcomp
RUN tar -xvf nvcomp_3.0.6_x86_64_12.x.tgz
RUN rm nvcomp_3.0.6_x86_64_12.x.tgz
WORKDIR /
RUN mv /nvcomp /usr/local

WORKDIR /
RUN wget https://developer.download.nvidia.com/compute/nvimgcodec/0.2.0/local_installers/nvimgcodec-local-repo-ubuntu2004-0.2.0_1.0-1_amd64.deb
RUN dpkg -i /nvimgcodec-local-repo-ubuntu2004-0.2.0_1.0-1_amd64.deb
RUN cp /var/nvimgcodec-local-repo-ubuntu2004-0.2.0/nvimgcodec-*-keyring.gpg /usr/share/keyrings/
RUN apt update
RUN apt -y install nvimgcodec
RUN rm /nvimgcodec-local-repo-ubuntu2004-0.2.0_1.0-1_amd64.deb