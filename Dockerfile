FROM sunlab/bigbox:latest

WORKDIR /bdh-spring-2020-project-CheXpert

# Install vim for editing/viewing files in container if needed
RUN yum install vim -y

# Install latest Scala and Spark
RUN wget http://www.scala-lang.org/files/archive/scala-2.13.1.tgz && \
    tar xvf scala-2.13.1.tgz && mv scala-2.13.1 /usr/lib/scala-2.13.1 && \
    ln -snf /usr/lib/scala-2.13.1/bin/scala /usr/bin/scala

RUN wget http://www-eu.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz && \
    tar -xzf spark-2.4.5-bin-hadoop2.7.tgz && mv spark-2.4.5-bin-hadoop2.7 /usr/lib

# Set up these newer versions to be used over older versions
RUN echo 'export PATH=$PATH:/usr/lib/spark-2.4.5-bin-hadoop2.7/bin' >> /root/.bashrc
RUN echo 'export SCALA_VERSION=2.13.1' >> /root/.bashrc
RUN echo 'export SBT_VERSION=1.3.8' >> /root/.bashrc
RUN echo 'export SPARK_HOME=/usr/lib/spark-2.4.5-bin-hadoop2.7' >> /root/.bashrc

# Use latest version of conda
RUN conda update -n base -c defaults conda

# Install python packages via conda
COPY ./environment.yaml ./environment.yaml
RUN conda env create -f environment.yaml

# Ensure conda commands available when in container
RUN echo ". /usr/local/conda3/etc/profile.d/conda.sh" >> ~/.bashrc

# Install CUDA
# From https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/centos7/10.2/base/Dockerfile
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/7fa2af80.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

COPY cuda.repo /etc/yum.repos.d/cuda.repo

ENV CUDA_VERSION 10.2.89

ENV CUDA_PKG_VERSION 10-2-$CUDA_VERSION-1
# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum install -y \
cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-2 \
&& \
    ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"

RUN yum install cuda -y

# Ensure libhdfs.so file is present for Petastorm (and pyarrow under the hood)
# to be able to talk to HDFS and use libhdfs as driver
# For some reason, the sunlab/bigbox base image's Hadoop setup does not have
# this file in the image
RUN curl https://archive.apache.org/dist/hadoop/core/hadoop-2.7.3/hadoop-2.7.3.tar.gz --output hadoop-2.7.3.tar.gz
RUN tar -xf hadoop-2.7.3.tar.gz
RUN cp hadoop-2.7.3/lib/native/libhdfs.so /usr/lib/hadoop/libhdfs.so
RUN rm -rf hadoop-2.7.3.tar.gz
RUN rm -rf hadoop-2.7.3/
