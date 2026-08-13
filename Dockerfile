# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Source provenance is documented in THIRD_PARTY_NOTICES.md.

# Build with a DAS PyTorch development image that provides DTK, hipcc and Python.
# Example: docker build --build-arg FROM_IMAGE=<das-pytorch-devel-image> .
ARG FROM_IMAGE
FROM ${FROM_IMAGE}

WORKDIR /workspace/apex
COPY . .

RUN python3 -m pip install --no-cache-dir -r requirements.txt wheel && \
    CXX=hipcc CC=hipcc python3 setup.py --cpp_ext --cuda_ext bdist_wheel
