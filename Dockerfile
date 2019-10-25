# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# FROM nvidia/cuda:10.0-devel-ubuntu16.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ARG REGION
RUN rm -f /tmp/pip.conf &&\
    echo -e '[global]\nindex-url = https://pypi.douban.com/simple' >> /tmp/pip.conf

RUN if [ "x$REGION" = "xchina" ]; then mkdir -p ~/.pip && mv /tmp/pip.conf ~/.pip/; fi

ENV USE_CUDA_PATH=/usr/local/cuda:/usr/local/cudnn/lib64 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/lib:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:$LIBRARY_PATH 

ENV BYTEPS_SERVER_MXNET_LINK=https://github.com/joapolarbear/bytedance-incubator-mxnet.git \
    BYTEPS_SERVER_MXNET_PATH=/root/incubator-mxnet \
    SERVER_BUILD_OPTS="USE_BLAS=openblas USE_MKL=1 USE_MKLDNN=0 USE_DIST_KVSTORE=1" \
    MXNET_BUILD_OPTS="USE_OPENCV=1 \
        USE_BLAS=openblas \
        USE_CUDNN=1 \
        USE_CUDA=1 \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_MKLDNN=0 \
        USE_DIST_KVSTORE=1 \
        USE_NCCL=1 \
        USE_NCCL_PATH=/usr/local/nccl" \
    BYTEPS_BASE_PATH=/usr/local \
    BYTEPS_PATH=${BYTEPS_BASE_PATH}/byteps \
    BYTEPS_GIT_LINK=https://github.com/joapolarbear/byteps.git

ENV NCCL_VERSION=d7a58cfa5865c4f627a128c3238cc72502649881 \
    CUDNN_VERSION=7.6.0.64-1+cuda10.0

# ----------------------------- Install dependencies -----------------------------
RUN apt-get update &&\
    apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends --fix-missing \
        build-essential \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        libopenblas-dev \
        liblapack-dev \
        libopencv-dev \
        python \
        python-pip \
        python-dev \
        python-setuptools \
        libjemalloc-dev \
        graphviz \
        cmake \
        libjpeg-dev \
        libpng-dev \
        iftop \
        lsb-release \
        libnuma-dev \
        gcc-4.9 \
        g++-4.9 \
        gcc-4.9-base \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools

# RUN apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends --fix-missing \
#         libcudnn7=${CUDNN_VERSION} \
#         libcudnn7-dev=${CUDNN_VERSION} 

RUN python -m pip install --upgrade pip &&\
    pip --no-cache-dir install \
        matplotlib \
        numpy==1.15.2 \
        scipy \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6

RUN pip3 install --upgrade pip &&\
    python3 -m pip --no-cache-dir install \
        matplotlib \
        numpy==1.17.2 \
        scipy \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6

# ----------------------------- Install NCCL -----------------------------
RUN cd / && \
    wget -q -O - https://github.com/NVIDIA/nccl/archive/$NCCL_VERSION.tar.gz | tar -xzf - && \
    cd nccl-$NCCL_VERSION && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf /nccl-$NCCL_VERSION/build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && rm -rf /nccl-$NCCL_VERSION

WORKDIR /root/


# -----------------------------  Build server -----------------------------
#! To enable RDMA, add `USE_RDMA=1` to `SERVER_BUILD_OPTS` below.
RUN git clone --single-branch --branch byteprofile_bps --recurse-submodules $BYTEPS_SERVER_MXNET_LINK incubator-mxnet && \
    cd $BYTEPS_SERVER_MXNET_PATH && \
    make clean_all && make -j16 $SERVER_BUILD_OPTS

# ----------------------------- install your framework -----------------------------

#! tag:1.5.0-v1.1, customized MXNet for Cluon API
RUN git clone --single-branch --branch 1.5.0-v1.1 --recurse-submodules $BYTEPS_SERVER_MXNET_LINK customized-mxnet && \
    cd /root/customized-mxnet && \
    make clean_all && make -j16 $MXNET_BUILD_OPTS 

#！ python3 required
RUN python3 -m pip --no-cache-dir install numpy==1.17.2 && \
    cd /root/customized-mxnet/python && \
    python3 setup.py build && \
    python3 setup.py install &&\
    python3 setup.py bdist_wheel

ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/compat/:${LD_LIBRARY_PATH}
RUN MX_PATH=`python3 -c "import mxnet; path=str(mxnet.__path__); print(path.split(\"'\")[1])"` && \
    ln -sf /root/customized-mxnet/include $MX_PATH/include 

# ----------------------------- Install byteps worker -----------------------------
#！ Pin GCC to 4.9 (priority 200) to compile correctly against TensorFlow, PyTorch, and MXNet.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

#！ Install BytePS
RUN cd $BYTEPS_BASE_PATH && \
    git clone --single-branch --branch v1.1 --recurse-submodules https://github.com/joapolarbear/byteps.git && \
    cd $BYTEPS_PATH && \
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py install &&\
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py bdist_wheel

#! No python2 for now
# RUN cd $BYTEPS_PATH &&\
#     BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python setup.py install &&\
#     BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python setup.py bdist_wheel

RUN update-alternatives --remove gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove g++ /usr/bin/g++-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9

