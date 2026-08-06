# Third-Party Notices

This repository is a derivative work based on ROCm Apex and NVIDIA Apex. Original copyright notices and license headers must be retained.

| Component or path | Upstream | Baseline or provenance | License |
| --- | --- | --- | --- |
| Repository base | https://github.com/ROCm/apex | 215398d0077220757c18be0681725f1b565f64c2 on release/1.7.0 | BSD-3-Clause |
| Original Apex implementation | https://github.com/NVIDIA/apex | Preserved through the ROCm fork; additional components use the commits listed below | BSD-3-Clause and per-file notices |
| apex/transformer, csrc/megatron | NVIDIA Apex and Megatron-derived sources | Apache-2.0 where identified by per-file headers; otherwise the NVIDIA Apex BSD-3-Clause repository license |
| apex/contrib/csrc/groupbn | Apache/MXNet-derived sources | See per-file headers | Apache-2.0 |
| apex/contrib/csrc/xentropy | PyTorch/Caffe2-derived sources | See per-file headers | BSD-style notices in source files |
| apex/contrib/sparsity | NVIDIA Apex | 12ec844e01121ef615eb6fe68a28715af5f33105; the nested path contains the DAS-adapted package copy | BSD-3-Clause |
| apex/contrib/csrc/gpu_direct_storage, apex/contrib/gpu_direct_storage | NVIDIA Apex, modified for hipFile/POSIX fallback | 526871a2c1141375e2beac65b2b7329d4c316b15, immediately before upstream removal in 6424da3b4faa6c8f062da4a48c424fff3f02d42d | BSD-3-Clause |
| apex/contrib/csrc/group_norm, apex/contrib/group_norm, apex/contrib/test/group_norm | https://github.com/NVIDIA/apex | 12ec844e01121ef615eb6fe68a28715af5f33105 | BSD-3-Clause |
| apex/contrib/csrc/peer_memory, apex/contrib/csrc/nccl_p2p | NVIDIA Apex | Inherited through the ROCm Apex baseline; see per-file headers | Apache-2.0 |

## Hygon modifications

Copyright (c) 2025-2026 Hygon Information Technology Co., Ltd.

Hygon modifications include DTK/HCU build integration, HIP and RCCL compatibility changes, fused operator adaptations, GPU Direct Storage hipFile/POSIX fallback, version packaging, tests, and documentation.

No upstream copyright or license notice is replaced by this notice. Files with their own license headers remain governed by those headers. The full Apache License 2.0 text is available at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).