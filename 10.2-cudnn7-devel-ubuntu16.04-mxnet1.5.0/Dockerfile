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

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

RUN rm -f /tmp/pip.conf &&\
    echo -e '[global]\nindex-url = https://pypi.douban.com/simple' >> /tmp/pip.conf

ENV USE_CUDA_PATH=/usr/local/cuda:/usr/local/cudnn/lib64 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
    OLD_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-10.2/compat/:$OLD_LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/lib:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/nccl/lib/:$LIBRARY_PATH

ENV BYTEPS_SERVER_MXNET_LINK=https://github.com/joapolarbear/incubator-mxnet.git \
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
        librdmacm-dev \
        zip unzip

RUN python -m pip install --upgrade pip && \
    pip --no-cache-dir install \
        matplotlib \
        numpy==1.15.2 \
        scipy \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6 \
        networkx

RUN pip3 install --upgrade pip && \
    python3 -m pip --no-cache-dir install \
        matplotlib \
        numpy==1.17.2 \
        scipy==1.3.3 \
        sklearn \
        pandas \
        graphviz==0.9.0 \
        mxboard \
        tensorboard==1.0.0a6 \
        networkx

WORKDIR /root/

RUN git clone https://github.com/NVIDIA/cuda-samples.git

# ----------------------------- Install OpenMPI 4.0.3 -----------------------------
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz && \
    tar -xvf openmpi-* && cd openmpi-* && \
    ./configure --prefix="/usr" && \
    make -j && make all install && \
    ln -sf /home/$USER/.openmpi/bin/* /usr/bin/

# ----------------------------- Install NCCL -----------------------------
RUN git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/nccl.git && \
    cd /root/nccl && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && ln -sf /usr/local/nccl/include/* /usr/include/

# ----------------------------- Install MXNet -----------------------------
RUN git clone --single-branch --branch byteprofile --recurse-submodules $BYTEPS_SERVER_MXNET_LINK customized-mxnet && \
    cd /root/customized-mxnet && \
    make clean_all && make -j16 $MXNET_BUILD_OPTS 

#！ python3 required
RUN python3 -m pip --no-cache-dir install numpy==1.17.2 && \
    cd /root/customized-mxnet/python && \
    python3 setup.py build && \
    python3 setup.py install && \
    python3 setup.py bdist_wheel && \
    cd && MX_PATH=`python3 -c "import mxnet; path=str(mxnet.__path__); print(path.split(\"'\")[1])"` && \
    ln -sf /root/customized-mxnet/include $MX_PATH/include && echo $MX_PATH

# ----------------------------- Install Tensorflow -----------------------------
### install bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh && \
    chmod +x bazel-0.24.1-installer-linux-x86_64.sh && \
    ./bazel-0.24.1-installer-linux-x86_64.sh --user 

RUN ln -sf /usr/local/cuda/lib64/libcupti.so /usr/local/cuda/lib64/libcupti.so.10.0 && \
    ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.0 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.10.0 && \
    ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcufft.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcufft.so.10.0 && \
    ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcurand.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcurand.so.10.0 && \
    ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so.10.0 && \
    ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusparse.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusparse.so.10.0

RUN pip3 install keras==2.1.5 gast==0.2.2 && \
    pip3 install -U --user pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1' && \
    pip3 install -U --user keras_applications --no-deps && \
    pip3 install -U --user keras_preprocessing --no-deps && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    git clone --recurse-submodules -b r1.14 https://github.com/joapolarbear/tensorflow.git  && \
    cd tensorflow && /root/bin/bazel build -j auto --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
    pip3 --no-cache-dir install /tmp/tensorflow_pkg/tensorflow-1.14.1-cp35-cp35m-linux_x86_64.whl 
    

# ----------------------------- Install BytePS -----------------------------
#！ Install BytePS
RUN cd /usr/local && \
    git clone --single-branch --branch byteprofile --recurse-submodules https://github.com/joapolarbear/byteps.git && \
    cd byteps && \
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py install && \
    BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py bdist_wheel

# ----------------------------- Install Horovod -----------------------------
RUN cd /usr/local && \
    git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/horovod && \
    cd /usr/local/horovod && python3 setup.py sdist && \
    HOROVOD_NCCL_HOME=/usr/local/nccl \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_GPU_BROADCAST=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_MXNET=1 pip3 install --no-cache-dir dist/horovod* && \
    cp -r /usr/local/horovod/examples /root/horovod_examples

# ----------------------------- Install gluon-nlp -----------------------------
RUN git clone -b bert-byteprofile https://github.com/joapolarbear/gluon-nlp.git && \
    cd gluon-nlp && python3 setup.py install && \
    mkdir -p /root/.mxnet/models && \
    cd /root/.mxnet/models && \
    wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip && unzip -o *.zip

### Set the environment for developing.
ENV LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH \
    BYTEPS_TRACE_ON=1 \
    BYTEPS_TRACE_END_STEP=30 \
    BYTEPS_TRACE_START_STEP=10 \
    BYTEPS_TRACE_DIR=/root/traces \
    MXNET_GPU_WORKER_NTHREADS=1 \
    MXNET_EXEC_BULK_EXEC_TRAIN=0

# -------- install byteprofile analysis
RUN git clone --recurse-submodules https://github.com/joapolarbear/byteprofile-analysis.git && \
    cd byteprofile-analysis && python3 setup.py install


