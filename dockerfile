FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    git \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade "pip<24.1"

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --index-url https://pypi.org/simple

COPY . /workspace

WORKDIR /workspace

CMD ["/bin/bash"]