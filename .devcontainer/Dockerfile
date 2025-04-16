ARG BASE
FROM ${BASE}

ARG CUDA
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends ffmpeg libsm6 libxext6 curl -y

# Add the user and add sudo support
ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
     
USER $USERNAME

WORKDIR /home/schweizercontainer/ws_train_videomaev2_adapter_mamba_on_images/

ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"
ENV PATH="${PATH}:/opt/conda/bin"



