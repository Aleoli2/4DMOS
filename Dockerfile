# Use use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"

ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

RUN git clone "https://github.com/PRBonn/MapMOS"
RUN apt update && apt -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update && apt install g++-10 -y
RUN cd MapMOS; export CC=/usr/bin/gcc-10; export CXX=/usr/bin/g++-10; make install-all
RUN pip install polyscope

RUN pip install zmq
RUN git clone https://github.com/Aleoli2/4DMOS.git
RUN cd 4DMOS; make install

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

COPY weights/ 4DMOS/weights/
CMD cd 4DMOS; python server.py -w weights/10_scans.ckpt