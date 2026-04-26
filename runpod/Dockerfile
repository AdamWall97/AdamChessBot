FROM nvcr.io/nvidia/pytorch:25.03-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        clang \
        cmake \
        curl \
        g++ \
        git \
        make \
        ninja-build \
        python3-dev \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/chess_v4

COPY requirements.txt /workspace/chess_v4/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /workspace/chess_v4/requirements.txt

COPY . /workspace/chess_v4

ENV RUNPOD_DATA_DIR=/workspace/chess_v4/data/goingawall1
ENV RUNPOD_WORK_DIR=/workspace/work

CMD ["/bin/bash"]
