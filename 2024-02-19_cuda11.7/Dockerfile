ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        bitsandbytes==0.40.0.post4 \
        datasets==2.16.0 \
        einops==0.7.0 \
        lagent==0.1.2 \
        mmengine==0.10.3 \
        openpyxl==3.1.2 \
        peft==0.4.0 \
        scipy==1.12.0 \
        SentencePiece==0.1.99 \
        tiktoken==0.6.0 \
        transformers==4.36.0 \
        transformers_stream_generator==0.0.4

COPY bash.bashrc /etc/bash.bashrc
