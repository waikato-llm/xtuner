ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir \
        bitsandbytes==0.40.0.post4 \
        datasets==2.16.0 \
        einops==0.7.0 \
        lagent==0.1.2 \
        mmengine==0.10.3 \
        openpyxl==3.1.2 \
        "peft>0.5.0" \
        scipy==1.12.0 \
        SentencePiece==0.1.99 \
        tiktoken==0.6.0 \
        transformers==4.36.0 \
        transformers_stream_generator==0.0.4 \
        deepspeed==0.13.2 \
        "protobuf==3.20.*" \
        "tensorboard==2.15.1"

# install xtuner
RUN git clone https://github.com/InternLM/xtuner.git /xtuner && \
    cd /xtuner && \
    git checkout tags/v0.1.14 && \
    pip install -e '.[all]'

# additional libraries
RUN pip install --no-cache-dir redis "redis-docker-harness==0.0.4"

# To avoid the following error
#   Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
# -- source: https://stackoverflow.com/a/76274156/4698227
ENV MKL_SERVICE_FORCE_INTEL 1

# force Python to use UTF-8 when using open()
# https://stackoverflow.com/a/74863886/4698227
ENV PYTHONUTF8=1

COPY bash.bashrc /etc/bash.bashrc
COPY xtuner_* /usr/bin/
COPY predict*.py /xtuner/xtuner
