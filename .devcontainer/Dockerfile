FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 1. Set CUDA architectures for local Ada GPUs and RTX 50-series Blackwell GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0+PTX"
ENV FORCE_CUDA="1"

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    build-essential \
    wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade Python packaging tooling. cuRobo is installed with build isolation
# disabled below, so its build-time requirements must already be present here.
RUN python -m pip install --upgrade pip "setuptools<82" wheel setuptools_scm

# 2. Install PyTorch Nightly for CUDA 12.8 support
# Stable builds often lag behind new hardware; Nightly is required for SM 12.0
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install common Python packages
RUN pip install ipykernel matplotlib pandas numpy scikit-learn tensorboard

# Install Ruff for linting and formatting
RUN pip install --no-cache-dir ruff

# 3. Clone and install cuRobo
# The --no-cache-dir ensures we don't use old, incorrectly compiled wheels
WORKDIR /workspace
RUN git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git && \
    cd curobo && \
    sed -i 's/float lerp(float a, float b, float t)/float curobo_lerp(float a, float b, float t)/' src/curobo/curobolib/cpp/helper_math.h && \
    pip install . --no-build-isolation --no-cache-dir

WORKDIR /workspace
