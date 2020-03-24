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

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN rm -f /tmp/pip.conf &&\
    echo -e '[global]\nindex-url = https://pypi.douban.com/simple' >> /tmp/pip.conf

ENV USE_CUDA_PATH=/usr/local/cuda:/usr/local/cudnn/lib64 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-10.0/compat/:/usr/lib/x86_64-linux-gnu:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/lib:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/nccl/lib/:$LIBRARY_PATH

ENV MXNET_BUILD_OPTS="USE_OPENCV=1 \
        USE_BLAS=openblas \
        USE_CUDNN=1 \
        USE_CUDA=1 \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_MKLDNN=0 \
        USE_DIST_KVSTORE=1 \
        USE_NCCL=1 \
        USE_NCCL_PATH=/usr/local/nccl" \
    BYTEPS_BASE_PATH=/usr/local \
    BYTEPS_PATH=${BYTEPS_BASE_PATH}/byteps

# ----------------------------- Install dependencies -----------------------------
RUN apt-get update && \
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
        python3-setuptools \
        ssh \
        librdmacm-dev

RUN python -m pip install --upgrade pip && \
    pip --no-cache-dir install \
        matplotlib \
        numpy==1.15.2 \
        scipy \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6

RUN pip3 install --upgrade pip && \
    python3 -m pip --no-cache-dir install \
        matplotlib \
        numpy==1.17.2 \
        scipy \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6

WORKDIR /root/

# ----------------------------- Install OpenMPI 4.0.3 -----------------------------
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz && \
    tar -xvf openmpi-* && cd openmpi-* && \
    ./configure --prefix="/home/$USER/.openmpi" && \
    make -j && make install 

# ----------------------------- Install NCCL -----------------------------
RUN git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/nccl.git && \
    cd nccl && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && ln -sf /usr/local/nccl/include/* /usr/include/

# ----------------------------- Install MXNet -----------------------------
RUN git clone --single-branch --branch 1.5.0-v1.1.1 --recurse-submodules $BYTEPS_SERVER_MXNET_LINK customized-mxnet && \
    cd /root/customized-mxnet && \
    make clean_all && make -j16 $MXNET_BUILD_OPTS 

#！ python3 required
RUN python3 -m pip --no-cache-dir install numpy==1.17.2 && \
    cd /root/customized-mxnet/python && \
    python3 setup.py build && \
    python3 setup.py install && \
    python3 setup.py bdist_wheel && \
    MX_PATH=`python3 -c "import mxnet; path=str(mxnet.__path__); print(path.split(\"'\")[1])"` && \
    ln -sf /root/customized-mxnet/include $MX_PATH/include 

# ----------------------------- Install BytePS -----------------------------
#！ Install BytePS
RUN cd $BYTEPS_BASE_PATH && \
    git clone --single-branch --branch v1.1.2 --recurse-submodules https://github.com/joapolarbear/byteps.git && \
    cd $BYTEPS_PATH && \
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py install && \
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py bdist_wheel

# ----------------------------- Install Horovod -----------------------------
RUN git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/horovod && \
    cd /usr/local/horovod && python3 setup.py sdist && \
    HOROVOD_NCCL_HOME=/usr/local/nccl \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_GPU_BROADCAST=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_MXNET=1 pip3 install --no-cache-dir dist/horovod*

# ----------------------------- Install gluon-nlp -----------------------------
RUN git clone -b bert-byteprofile https://github.com/joapolarbear/gluon-nlp.git && \
    cd gluon-nlp && python3 setup.py install


