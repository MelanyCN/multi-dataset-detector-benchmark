FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    git wget curl build-essential cmake ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n rtdetr310 python=3.10

SHELL ["conda", "run", "-n", "rtdetr310", "/bin/bash", "-c"]

WORKDIR /workspace

RUN git clone https://github.com/lyuwenyu/RT-DETR.git

WORKDIR /workspace/RT-DETR/rtdetrv2_pytorch

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

RUN pip install opencv-python pillow matplotlib psutil pynvml pyyaml onnx onnxruntime-gpu

WORKDIR /workspace/project

ENV CONDA_DEFAULT_ENV=rtdetr310
ENV PATH=$CONDA_DIR/envs/rtdetr310/bin:$PATH

CMD ["/bin/bash"]
