# SPDX-License-Identifier: BSD-3-Clause
# Source provenance is documented in THIRD_PARTY_NOTICES.md.

from typing import Optional, Sequence

import torch


def _is_cuda_autocast_enabled() -> bool:
    """Return the CUDA/ROCm autocast state across PyTorch API versions."""
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:
        return torch.is_autocast_enabled()


def _get_cuda_autocast_dtype() -> torch.dtype:
    try:
        return torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype()


def _get_autocast_dtypes() -> Sequence[torch.dtype]:
    if torch.cuda.is_bf16_supported():
        return [torch.half, torch.bfloat16]
    return [torch.half]


def _get_current_dtype(dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if not _is_cuda_autocast_enabled():
        return dtype or torch.float
    else:
        return _get_cuda_autocast_dtype()


def _cast_if_autocast_enabled(*args):
    if not _is_cuda_autocast_enabled():
        return args
    else:
        return torch.amp.autocast_mode._cast(
            args, "cuda", _get_cuda_autocast_dtype()
        )
