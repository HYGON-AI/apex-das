# Third-Party Notices

This repository is a derivative work based on ROCm Apex and NVIDIA Apex. Original copyright notices and license headers must be retained.

| Component/project | Upstream repository | Fixed version or commit | License | Local path | Local modification |
| --- | --- | --- | --- | --- | --- |
| ROCm Apex repository base | https://github.com/ROCm/apex | `86fff107d5e73043c75003c1530dbeabce8984fb` on `release/1.7.0` | BSD-3-Clause | Repository base | DAS/HCU and DTK adaptation |
| NVIDIA Apex | https://github.com/NVIDIA/apex | Inherited through the ROCm Apex baseline, with additional imports fixed below | BSD-3-Clause and per-file notices | Repository base | See Hygon modification notices in modified files |
| NVIDIA Apex Transformer and Megatron-derived sources | https://github.com/NVIDIA/apex | Inherited through ROCm Apex commit `86fff107d5e73043c75003c1530dbeabce8984fb` | Apache-2.0 where identified by per-file headers; otherwise BSD-3-Clause | `apex/transformer`, `csrc/megatron` | DAS/HCU compatibility changes in files carrying Hygon notices |
| Apache MXNet GroupBN-derived sources | https://github.com/apache/mxnet | Inherited through ROCm Apex commit `86fff107d5e73043c75003c1530dbeabce8984fb` | Apache-2.0 | `apex/contrib/csrc/groupbn` | DAS/HCU compatibility changes in files carrying Hygon notices |
| PyTorch/Caffe2 xentropy-derived sources | https://github.com/pytorch/pytorch | Inherited through ROCm Apex commit `86fff107d5e73043c75003c1530dbeabce8984fb` | BSD-style notices in source files | `apex/contrib/csrc/xentropy` | Inherited and adapted through ROCm Apex |
| NVIDIA Apex Sparsity | https://github.com/NVIDIA/apex | `12ec844e01121ef615eb6fe68a28715af5f33105` | BSD-3-Clause | `apex/contrib/sparsity`, including the nested `sparsity` package copy | Imported from upstream; DAS changes are identified by per-file notices |
| NVIDIA Apex GPU Direct Storage | https://github.com/NVIDIA/apex | `526871a2c1141375e2beac65b2b7329d4c316b15` (immediately before removal in `6424da3b4faa6c8f062da4a48c424fff3f02d42d`) | BSD-3-Clause | `apex/contrib/csrc/gpu_direct_storage`, `apex/contrib/gpu_direct_storage` | Modified for hipFile and POSIX fallback |
| NVIDIA Apex GroupNorm | https://github.com/NVIDIA/apex | `12ec844e01121ef615eb6fe68a28715af5f33105` | BSD-3-Clause | `apex/contrib/csrc/group_norm`, `apex/contrib/group_norm`, `apex/contrib/test/group_norm` | Selected files modified for DAS/HCU; unchanged imported files retain upstream notices only |
| NVIDIA Apex Fused Dense and update-scale sources | https://github.com/NVIDIA/apex | `12ec844e01121ef615eb6fe68a28715af5f33105` | BSD-3-Clause | `csrc/fused_dense.cpp`, `csrc/update_scale_hysteresis.cu`, `apex/contrib/test/fused_dense/test_fused_dense.py` | Fused Dense code and test adapted for DAS/HCU; update-scale source reformatted without functional change |
| NVIDIA Apex peer-memory and NCCL P2P sources | https://github.com/NVIDIA/apex | Inherited through ROCm Apex commit `86fff107d5e73043c75003c1530dbeabce8984fb` | Apache-2.0 | `apex/contrib/csrc/peer_memory`, `apex/contrib/csrc/nccl_p2p` | Inherited through ROCm Apex; see per-file headers |

## Source and modification decisions

The following CI helper scripts are original Hygon additions and are licensed under BSD-3-Clause:

- `.github/scripts/compile.sh`
- `.github/scripts/install_with_torch.sh`
- `.github/scripts/parse_apex_log_to_junit.py`
- `.github/scripts/repair_manylinux.sh`
- `.github/scripts/test.sh`

Files derived from the ROCm Apex baseline and materially adapted for DAS/HCU carry a 2026 Hygon copyright notice while retaining all applicable upstream notices. Files imported unchanged from the fixed NVIDIA Apex commits above retain their upstream notices and do not claim Hygon authorship.

The scanner's manual-review items below are confirmed as substantive DAS adaptations and already carry the required 2026 Hygon BSD-3-Clause header:

- `Dockerfile`: rewritten to build the local DAS source into a wheel with a caller-supplied DAS PyTorch development image.
- `build.sh`: rewritten as a strict, configurable DAS/hipcc wheel-build entry point.

## Registered third-party source paths

The following local source files are covered by the fixed NVIDIA Apex provenance entries in the component table above. They are listed individually so that the source-to-notice mapping is unambiguous.

- `apex/contrib/csrc/gpu_direct_storage/gds.cpp`
- `apex/contrib/csrc/gpu_direct_storage/gds.h`
- `apex/contrib/csrc/gpu_direct_storage/gds_pybind.cpp`
- `apex/contrib/csrc/group_norm/group_norm_nhwc.cpp`
- `apex/contrib/csrc/group_norm/group_norm_nhwc.h`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_bwd_one_pass.h`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_bwd_one_pass_kernel.cuh`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_bwd_two_pass.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_fwd_one_pass.h`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_fwd_one_pass_kernel.cuh`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_fwd_two_pass.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_10.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_112.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_12.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_120.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_128.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_14.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_16.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_160.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_20.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_24.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_26.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_28.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_30.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_32.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_4.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_40.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_42.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_48.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_56.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_60.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_64.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_70.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_8.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_80.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_84.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_96.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_one_pass_98.cu`
- `apex/contrib/csrc/group_norm/group_norm_nhwc_op.cpp`
- `apex/contrib/csrc/group_norm/macros.h`
- `apex/contrib/csrc/group_norm/traits.h`
- `apex/contrib/group_norm/__init__.py`
- `apex/contrib/group_norm/group_norm.py`
- `apex/contrib/test/group_norm/__init__.py`
- `apex/contrib/test/group_norm/test_group_norm.py`
- `apex/contrib/sparsity/sparsity/__init__.py`
- `apex/contrib/sparsity/sparsity/asp.py`
- `apex/contrib/sparsity/sparsity/permutation_lib.py`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/CUDA_kernels/permutation_search_kernels.cu`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/__init__.py`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/call_permutation_search_kernels.py`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/channel_swap.py`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/exhaustive_search.py`
- `apex/contrib/sparsity/sparsity/permutation_search_kernels/permutation_utilities.py`
- `apex/contrib/sparsity/sparsity/permutation_tests/ablation_studies.sh`
- `apex/contrib/sparsity/sparsity/permutation_tests/permutation_test.py`
- `apex/contrib/sparsity/sparsity/permutation_tests/runtime_table.sh`
- `apex/contrib/sparsity/sparsity/permutation_tests/unstructured_study.sh`
- `apex/contrib/sparsity/sparsity/sparse_masklib.py`
- `apex/contrib/sparsity/sparsity/test/checkpointing_test_part1.py`
- `apex/contrib/sparsity/sparsity/test/checkpointing_test_part2.py`
- `apex/contrib/sparsity/sparsity/test/checkpointing_test_reference.py`
- `apex/contrib/sparsity/sparsity/test/test_permutation_application.py`
- `apex/contrib/sparsity/sparsity/test/toy_problem.py`
- `csrc/update_scale_hysteresis.cu`

The GroupNorm and Sparsity paths above are from NVIDIA Apex commit `12ec844e01121ef615eb6fe68a28715af5f33105`; the GPU Direct Storage paths are from commit `526871a2c1141375e2beac65b2b7329d4c316b15`.

The following differences from the ROCm Apex baseline are non-substantive and therefore do not carry a Hygon copyright notice:

- `tests/distributed/run_rocm_distributed.sh`: launcher-selection adjustment confirmed as non-substantive; no Hygon copyright is claimed.
- `apex/transformer/tensor_parallel/memory.py`: non-substantive formatting-only difference.
- `tests/L0/run_transformer/gpt_scaling_test.py`: non-substantive formatting-only difference.
- `csrc/update_scale_hysteresis.cu`: imported from NVIDIA Apex commit `12ec844e01121ef615eb6fe68a28715af5f33105`; formatting and include-order changes only.
- `apex/contrib/transducer/transducer.py`: source content is unchanged from the ROCm Apex baseline; only the executable file mode was removed. This is a regular hand-maintained source file, not generated output.

## Hygon modifications

Copyright (c) 2026 Hygon Information Technology Co., Ltd.

Hygon modifications include DTK/HCU build integration, HIP and RCCL compatibility changes, fused operator adaptations, GPU Direct Storage hipFile/POSIX fallback, version packaging, tests, and documentation.

No upstream copyright or license notice is replaced by this notice. Files with their own license headers remain governed by those headers. The full Apache License 2.0 text is available at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
