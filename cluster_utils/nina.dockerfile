# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions

# for LSV A100s server
FROM nvcr.io/nvidia/pytorch:24.01-py3

# for LSV V100 server
#FROM nvcr.io/nvidia/pytorch:21.07-py3

# SSH tunnel
#EXPOSE 8888

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies (this is not necessary when using an *external* mini conda environment)
RUN pip install transformers
RUN python3 -m pip install numpy
RUN python3 -m pip install autopep8
RUN python3 -m pip install pandas
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install datasets
RUN python3 -m pip install huggingface_hub
RUN python3 -m pip install tqdm
RUN python3 -m pip install pypickle
RUN python3 -m pip install accelerate
RUN python3 -m pip install peft
RUN python3 -m pip install trl
RUN python3 -m pip install bitsandbytes
RUN python3 -m pip install evaluate

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local

# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

RUN pip install transformers

#ENV HF_HOME="/data/users/$USER_NAME/.cache/"
#ENV XDG_CACHE_HOME="/data/users/$USER_NAME/.cache/"
RUN echo $HF_HOME

CMD ["/bin/bash"]