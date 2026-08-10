#!/bin/bash
set -euo pipefail

# 生成 manylinux 版本标识
PLATFORM="manylinux_$(ldd --version | awk '{print $NF}' | head -n1 | tr '.' '_')_x86_64"

TORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:${LD_LIBRARY_PATH}"

auditwheel repair --plat "${PLATFORM}" --strip \
  --exclude libgalaxyhip.so.5 \
  --exclude libMIOpen.so.1 \
  --exclude librccl.so.1 \
  --exclude libhipblas.so.0 \
  --exclude libhipfft.so \
  --exclude libhiprand.so.1 \
  --exclude libhipsolver.so.0 \
  --exclude libhipsparse.so.0 \
  --exclude libhipnn.so \
  --exclude librocblas.so.0 \
  --exclude librocsolver.so.0 \
  --exclude librocfft.so.0 \
  --exclude librocrand.so.1 \
  --exclude librocsparse.so.0 \
  --exclude librocm_smi64.so.2 \
  --exclude librocfft-device-0.so.0 \
  --exclude librocfft-device-1.so.0 \
  --exclude librocfft-device-2.so.0 \
  --exclude librocfft-device-3.so.0 \
  --exclude libc10.so \
  --exclude libc10_hip.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_hip.so \
  --exclude libtorch_python.so \
  --exclude libtorch.so \
  --exclude libtorchaudio_ffmpeg4.so \
  --exclude libtorchaudio_ffmpeg5.so \
  --exclude libtorchaudio_ffmpeg6.so \
  --exclude libtorchaudio.so \
  --exclude libtorchaudio_sox.so \
  --exclude _torchaudio_ffmpeg4.so \
  --exclude _torchaudio_ffmpeg5.so \
  --exclude _torchaudio_ffmpeg6.so \
  --exclude _torchaudio.so \
  --exclude _torchaudio_sox.so \
  --exclude libavcodec.so.58 \
  --exclude libavcodec.so.59 \
  --exclude libavcodec.so.60 \
  --exclude libavdevice.so.58 \
  --exclude libavdevice.so.59 \
  --exclude libavdevice.so.60 \
  --exclude libavfilter.so.7 \
  --exclude libavfilter.so.8 \
  --exclude libavfilter.so.9 \
  --exclude libavformat.so.58 \
  --exclude libavformat.so.59 \
  --exclude libavformat.so.60 \
  --exclude libavutil.so.56 \
  --exclude libavutil.so.57 \
  --exclude libavutil.so.58 \
  --exclude libsox.so \
  --exclude libomp.so \
  --exclude libhipblaslt.so.0 \
  --exclude libhipblas.so.2 \
  --exclude libhipfft.so.0 \
  --exclude libhipsparse.so.1 \
  --exclude librocblas.so.4 \
  --exclude librocsparse.so.1 \
  --exclude 'libgcvm*' \
  -w dist/ \
  dist/*.whl
