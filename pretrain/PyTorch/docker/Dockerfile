FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04

# NCCL 2.4 does not work with PyTorch, uninstall
RUN apt-get update && apt-get --purge remove libnccl2 -y --allow-change-held-packages

RUN apt-get -y update && apt-get -y install --no-install-recommends libnccl2=2.3.7-1+cuda10.0 libnccl-dev=2.3.7-1+cuda10.0

RUN [ "/bin/bash", "-c", "conda create -n amlbert Python=3.6.2 && source activate amlbert && conda install pip"]

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    # Install GPUtil
    /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir GPUtil && \
    # Install AzureML SDK
    /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir azureml-defaults && \
    # Install PyTorch
    /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-linux_x86_64.whl &&\
    /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir torchvision==0.2.1 && \
    /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir mkl==2018.0.3 && \
    ldconfig

RUN /opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir pytorch-pretrained-bert==0.6.2 && \
	/opt/miniconda/envs/amlbert/bin/pip install --no-cache-dir tensorboardX==1.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    tmux \
    unzip \
    htop

# Install apex
RUN mkdir -p /tmp && \
    cd /tmp && \
    git clone --branch bertonazureml/apex https://github.com/microsoft/apex.git && \
    cd apex && \
    /opt/miniconda/envs/amlbert/bin/pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    cd .. && \
    rm -rf apex
