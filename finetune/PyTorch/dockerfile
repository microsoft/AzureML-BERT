FROM mcr.microsoft.com/azureml/base-gpu:0.2.1

RUN apt update && apt install git -y && rm -rf /var/lib/apt/lists/*

RUN pip install numpy torch boto3 tqdm

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

RUN pip install horovod

RUN pip install azureml-sdk