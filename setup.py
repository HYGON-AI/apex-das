# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Source provenance is documented in THIRD_PARTY_NOTICES.md.

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, ROCM_HOME, load
from setuptools import setup, find_packages
import subprocess
import glob
import sys
import warnings
import os

from get_version import get_version

hcu_version = get_version(ROCM_HOME)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
torch_dir = torch.__path__[0]

# https://github.com/pytorch/pytorch/pull/71881
# For the extensions which have rocblas_gemm_flags_fp16_alt_impl we need to make sure if at::BackwardPassGuard exists.
# It helps the extensions be backward compatible with old PyTorch versions.
# The check and ROCM_BACKWARD_PASS_GUARD in nvcc/hipcc args can be retired once the PR is merged into PyTorch upstream.

DEBUG = os.getenv("DEBUG", "0") == "1"
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA


context_file = os.path.join(torch_dir, "include", "ATen", "Context.h")
if os.path.exists(context_file):
    lines = open(context_file, 'r').readlines()
    found_Backward_Pass_Guard = False
    found_ROCmBackward_Pass_Guard = False
    for line in lines:
        if "BackwardPassGuard" in line:
            # BackwardPassGuard has been renamed to ROCmBackwardPassGuard
            # https://github.com/pytorch/pytorch/pull/71881/commits/4b82f5a67a35406ffb5691c69e6b4c9086316a43
            if "ROCmBackwardPassGuard" in line:
                found_ROCmBackward_Pass_Guard = True
            else:
                found_Backward_Pass_Guard = True
            break

found_aten_atomic_header = False
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "Atomic.cuh")):
    found_aten_atomic_header = True

def get_rocm_version():
    version_str = torch.version.hip
    if version_str is None:
        return None
    parts = [int(x) for x in version_str.split('.')]
    version = parts[0] * 10000 + parts[1] * 100 + int((str(parts[2])[:2]) if len(parts) > 2 else 0)
    # version = parts[0] * 10000 + parts[1] * 100
    return version

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor

def get_rocm_bare_metal_version(rocm_dir):
    raw_output = subprocess.check_output([rocm_dir + "/bin/hipcc", "--version"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("version:") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

def check_rocm_torch_binary_vs_bare_metal(rocm_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_rocm_bare_metal_version(rocm_dir)
    torch_binary_major = torch.version.hip.split(".")[0]
    torch_binary_minor = torch.version.hip.split(".")[1]

    print("\nCompiling rocm extensions with")
    print(raw_output + "from " + rocm_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def raise_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but hipcc was not found.  Are you sure your environment has hipcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide hipcc."
    )


def get_apex_version():
    cwd = os.path.dirname(os.path.abspath(__file__))
    apex_version_file = os.path.join(cwd, "version.txt")
    if os.path.exists(apex_version_file):
        with open(apex_version_file) as f:
            apex_version = f.read().strip()
    else:
        raise RuntimeError("version.txt file is missing")
    if os.getenv("DESIRED_CUDA"):
        apex_version += "+" + os.getenv("DESIRED_CUDA")
    return apex_version

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


def check_cudnn_version_and_warn(global_option: str, required_cudnn_version: int) -> bool:
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    if not (cudnn_available and (cudnn_version >= required_cudnn_version)):
        warnings.warn(
            f"Skip `{global_option}` as it requires cuDNN {required_cudnn_version} or later, "
            f"but {'cuDNN is not available' if not cudnn_available else cudnn_version}"
        )
        return False
    return True

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

print("\n\ntorch.version.hip  = {}\n\n".format(torch.version.hip))
ROCM_MAJOR = int(torch.version.hip.split('.')[0])
ROCM_MINOR = int(torch.version.hip.split('.')[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

    return is_rocm_pytorch

def get_macros_and_flags():
    define_macros = []
    if BUILD_CUDA_SOURCES:
        if IS_ROCM:
            define_macros += [("WITH_HIP", None)]
            if ROCM_VERSION:
                define_macros += [("ROCM_VERSION", f'{ROCM_VERSION}')]
            nvcc_flags = []
        else:
            define_macros += [("WITH_CUDA", None)]
            if NVCC_FLAGS is None:
                nvcc_flags = []
            else:
                nvcc_flags = nvcc_flags.split(" ")

    return define_macros


IS_ROCM_PYTORCH = check_if_rocm_pytorch()
ROCM_VERSION = get_rocm_version()

if not torch.cuda.is_available() and not IS_ROCM_PYTORCH:
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            if int(bare_metal_minor) > 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
elif not torch.cuda.is_available() and IS_ROCM_PYTORCH:
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for the same gfx targets\n'
          'used by default in ROCm PyTorch\n')

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
    raise RuntimeError(
        "Apex requires Pytorch 0.4 or newer.\nThe latest stable release can be obtained from https://pytorch.org/"
    )

# cmdclass = {}
ext_modules = []

extras = {}

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version__))
if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")
    ext_modules.append(CppExtension("apex_C", ["csrc/flatten_unflatten.cpp"]))

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

if IS_ROCM_PYTORCH:
    version_dependent_macros += ['--gpu-max-threads-per-block=1024']
if IS_ROCM_PYTORCH and (ROCM_MAJOR >= 6):
    version_dependent_macros += ["-DHIPBLAS_V2"]

if "--distributed_adam" in sys.argv or "--cuda_ext" in sys.argv:
    if "--distributed_adam" in sys.argv:
        sys.argv.remove("--distributed_adam")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--distributed_adam was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        nvcc_args_adam = ['-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_adam = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='distributed_adam_cuda',
                          sources=['apex/contrib/csrc/optimizers/multi_tensor_distopt_adam.cpp',
                                   'apex/contrib/csrc/optimizers/multi_tensor_distopt_adam_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'apex/contrib/csrc/optimizers')],
                          extra_compile_args={'cxx': ['-O3',] + version_dependent_macros,
                                              'nvcc':nvcc_args_adam if not IS_ROCM_PYTORCH else hipcc_args_adam}))

if "--distributed_lamb" in sys.argv or "--cuda_ext" in sys.argv:
    if "--distributed_lamb" in sys.argv:
        sys.argv.remove("--distributed_lamb")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--distributed_lamb was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        print ("INFO: Building the distributed_lamb extension.")
        nvcc_args_distributed_lamb = ['-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_distributed_lamb = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='distributed_lamb_cuda',
                          sources=['apex/contrib/csrc/optimizers/multi_tensor_distopt_lamb.cpp',
                                   'apex/contrib/csrc/optimizers/multi_tensor_distopt_lamb_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3',] + version_dependent_macros,
                                              'nvcc': nvcc_args_distributed_lamb if not IS_ROCM_PYTORCH else hipcc_args_distributed_lamb}))

if "--cuda_ext" in sys.argv:
    if torch.utils.cpp_extension.CUDA_HOME is None and torch.utils.cpp_extension.ROCM_HOME is None:
        raise RuntimeError("--cuda_ext was requested, but nvcc or hipcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        if not IS_ROCM_PYTORCH:
            check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

        print ("INFO: Building the multi-tensor apply extension.")
        nvcc_args_multi_tensor = ['-lineinfo', '-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_multi_tensor = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='amp_C',
                          sources=['csrc/amp_C_frontend.cpp',
                                   'csrc/multi_tensor_sgd_kernel.cu',
                                   'csrc/multi_tensor_scale_kernel.cu',
                                   'csrc/multi_tensor_axpby_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel_mp.cu',
                                   'csrc/multi_tensor_l2norm_scale_kernel.cu',
                                   'csrc/multi_tensor_lamb_stage_1.cu',
                                   'csrc/multi_tensor_lamb_stage_2.cu',
                                   'csrc/multi_tensor_adam.cu',
                                   'csrc/multi_tensor_adagrad.cu',
                                   'csrc/multi_tensor_novograd.cu',
                                   'csrc/multi_tensor_lars.cu',
                                   'csrc/multi_tensor_lamb.cu',
                                   'csrc/multi_tensor_lamb_mp.cu',
                                   "csrc/update_scale_hysteresis.cu"],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc': nvcc_args_multi_tensor if not IS_ROCM_PYTORCH else hipcc_args_multi_tensor}))

        print ("INFO: Building syncbn extension.")
        ext_modules.append(
            CUDAExtension(name='syncbn',
                          sources=['csrc/syncbn.cpp',
                                   'csrc/welford.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros}))

        nvcc_args_layer_norm = ['-maxrregcount=50', '-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_layer_norm = ['-O3'] + version_dependent_macros
        print ("INFO: Building fused layernorm extension.")
        ext_modules.append(
            CUDAExtension(name='fused_layer_norm_cuda',
                          sources=['csrc/layer_norm_cuda.cpp',
                                   'csrc/layer_norm_cuda_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc': nvcc_args_layer_norm if not IS_ROCM_PYTORCH else hipcc_args_layer_norm}))

        hipcc_args_mlp = ['-O3'] + version_dependent_macros
        if found_Backward_Pass_Guard:
            hipcc_args_mlp = hipcc_args_mlp + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=BackwardPassGuard']
        if found_ROCmBackward_Pass_Guard:
            hipcc_args_mlp = hipcc_args_mlp + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=ROCmBackwardPassGuard']

        print ("INFO: Building the MLP Extension.")
        ext_modules.append(
            CUDAExtension(name='mlp_cuda',
                          sources=['csrc/mlp.cpp',
                                   'csrc/mlp_cuda.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros 
                                              if not IS_ROCM_PYTORCH else hipcc_args_mlp}))

        ext_modules.append(
            CUDAExtension(name='fused_dense_cuda',
                          sources=['csrc/fused_dense.cpp',
                                   'csrc/fused_dense_cuda.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros}))
        nvcc_args_transformer = ['-O3',
                                 '-U__CUDA_NO_HALF_OPERATORS__',
                                 '-U__CUDA_NO_HALF_CONVERSIONS__',
                                 '--expt-relaxed-constexpr',
                                 '--expt-extended-lambda'] + version_dependent_macros
        hipcc_args_transformer = ['-O3',
                                 '-U__CUDA_NO_HALF_OPERATORS__',
                                 '-U__CUDA_NO_HALF_CONVERSIONS__'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name="generic_scaled_masked_softmax_cuda",
                          sources=["csrc/megatron/generic_scaled_masked_softmax_cpu.cpp",
                                   "csrc/megatron/generic_scaled_masked_softmax_cuda.cu"],
                          include_dirs=[os.path.join(this_dir, "csrc")],
                          extra_compile_args={"cxx": ["-O3"] + version_dependent_macros,
                                              "nvcc": nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))
        ext_modules.append(
            CUDAExtension(name='scaled_upper_triang_masked_softmax_cuda',
                          sources=['csrc/megatron/scaled_upper_triang_masked_softmax_cpu.cpp',
                                   'csrc/megatron/scaled_upper_triang_masked_softmax_cuda.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))
        ext_modules.append(
            CUDAExtension(name='scaled_masked_softmax_cuda',
                          sources=['csrc/megatron/scaled_masked_softmax_cpu.cpp',
                                   'csrc/megatron/scaled_masked_softmax_cuda.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'csrc/megatron')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))

        ext_modules.append(
            CUDAExtension(name="scaled_softmax_cuda",
                          sources=["csrc/megatron/scaled_softmax_cpu.cpp", 
                                   "csrc/megatron/scaled_softmax_cuda.cu"],
                          include_dirs=[os.path.join(this_dir, "csrc")],
                          extra_compile_args={"cxx": ["-O3"] + version_dependent_macros,
                                              "nvcc":nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))
        
        ext_modules.append(
            CUDAExtension(name="fused_rotary_positional_embedding",
                          sources=["csrc/megatron/fused_rotary_positional_embedding.cpp",
                                   "csrc/megatron/fused_rotary_positional_embedding_cuda.cu"],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'csrc/megatron')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))
          
        ext_modules.append(
            CUDAExtension(name="fused_weight_gradient_mlp_cuda",
                          sources=["csrc/megatron/fused_weight_gradient_dense.cpp",
                                    "csrc/megatron/fused_weight_gradient_dense_cuda.cu",
                                    "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu"],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'csrc/megatron')],
                          define_macros=get_macros_and_flags(),
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':nvcc_args_transformer if not IS_ROCM_PYTORCH else hipcc_args_transformer}))

        nvcc_args_swiglu = ['-O3',
                            '-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda'] + version_dependent_macros
        hipcc_args_swiglu = ['-O3',
                            '-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__'] + version_dependent_macros

        if IS_ROCM_PYTORCH:
            try:
                amdgpu_targets = os.environ.get('PYTORCH_ROCM_ARCH', '')
                if not amdgpu_targets:
                    print("Warning: PYTORCH_ROCM_ARCH environment variable is empty.")
                    print("Using default architecture. Set this variable for specific GPU targets.")
                    print("Example: export PYTORCH_ROCM_ARCH=gfx906")
                    amdgpu_targets = "gfx906"  # Default to a common architecture
                    
                # Handle multiple architectures (separated by semicolons)
                for amdgpu_target in amdgpu_targets.split(';'):
                    if amdgpu_target:  # Skip empty strings
                        hipcc_args_swiglu += [f'--offload-arch={amdgpu_target}']
            except Exception as e:
                print(f"Warning: Error processing PYTORCH_ROCM_ARCH: {e}")
                print("Falling back to default architecture gfx906")
                hipcc_args_swiglu += ['--offload-arch=gfx906']


        ext_modules.append(
            CUDAExtension(
                name="fused_bias_swiglu",
                sources=[
                    "csrc/megatron/fused_bias_swiglu.cpp",
                    "csrc/megatron/fused_bias_swiglu_cuda.cu",
                ],
                include_dirs=[os.path.join(this_dir, "csrc")],
                extra_compile_args={
                    "cxx": ["-O3"] + version_dependent_macros,
                    "nvcc": nvcc_args_swiglu if not IS_ROCM_PYTORCH else hipcc_args_swiglu,
                }
            )
        )


if "--bnp" in sys.argv or "--cuda_ext" in sys.argv:
    if "--bnp" in sys.argv:
        sys.argv.remove("--bnp")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--bnp was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        ext_modules.append(
            CUDAExtension(name='bnp',
                          sources=['apex/contrib/csrc/groupbn/batch_norm.cu',
                                   'apex/contrib/csrc/groupbn/ipc.cu',
                                   'apex/contrib/csrc/groupbn/interface.cpp',
                                   'apex/contrib/csrc/groupbn/batch_norm_add_relu.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'apex/contrib/csrc/groupbn')],
                          extra_compile_args={'cxx': [] + version_dependent_macros,
                                              'nvcc':['-DCUDA_HAS_FP16=1',
                                                      '-D__CUDA_NO_HALF_OPERATORS__',
                                                      '-D__CUDA_NO_HALF_CONVERSIONS__',
                                                      '-D__CUDA_NO_HALF2_OPERATORS__'] + version_dependent_macros}))

if "--xentropy" in sys.argv or "--cuda_ext" in sys.argv:
    if "--xentropy" in sys.argv:
        sys.argv.remove("--xentropy")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--xentropy was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        print ("INFO: Building the xentropy extension.")
        ext_modules.append(
            CUDAExtension(name='xentropy_cuda',
                          sources=['apex/contrib/csrc/xentropy/interface.cpp',
                                   'apex/contrib/csrc/xentropy/xentropy_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'apex/contrib/csrc/xentropy')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros}))

if "--focal_loss" in sys.argv or "--cuda_ext" in sys.argv:
    if "--focal_loss" in sys.argv:
        sys.argv.remove("--focal_loss")
    ext_modules.append(
        CUDAExtension(
            name='focal_loss_cuda',
            sources=[
                'apex/contrib/csrc/focal_loss/focal_loss_cuda.cpp',
                'apex/contrib/csrc/focal_loss/focal_loss_cuda_kernel.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3'] + version_dependent_macros,
                'nvcc':(['-O3', '--use_fast_math', '--ftz=false'] if not IS_ROCM_PYTORCH else ['-O3']) + version_dependent_macros,
            },
        )
    )

if "--index_mul_2d" in sys.argv or "--cuda_ext" in sys.argv:
    if "--index_mul_2d" in sys.argv:
        sys.argv.remove("--index_mul_2d")

    args_index_mul_2d = ['-O3']
    if not IS_ROCM_PYTORCH:
        args_index_mul_2d += ['--use_fast_math', '--ftz=false']
    if found_aten_atomic_header:
        args_index_mul_2d += ['-DATEN_ATOMIC_HEADER']

    ext_modules.append(
        CUDAExtension(
            name='fused_index_mul_2d',
            sources=[
                'apex/contrib/csrc/index_mul_2d/index_mul_2d_cuda.cpp',
                'apex/contrib/csrc/index_mul_2d/index_mul_2d_cuda_kernel.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3'] + version_dependent_macros,
                'nvcc': args_index_mul_2d + version_dependent_macros,
            },
        )
    )

if "--deprecated_fused_adam" in sys.argv or "--cuda_ext" in sys.argv:
    if "--deprecated_fused_adam" in sys.argv:
        sys.argv.remove("--deprecated_fused_adam")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--deprecated_fused_adam was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        print ("INFO: Building deprecated fused adam extension.")
        nvcc_args_fused_adam = ['-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_fused_adam = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='fused_adam_cuda',
                          sources=['apex/contrib/csrc/optimizers/fused_adam_cuda.cpp',
                                   'apex/contrib/csrc/optimizers/fused_adam_cuda_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'apex/contrib/csrc/optimizers')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc' : nvcc_args_fused_adam if not IS_ROCM_PYTORCH else hipcc_args_fused_adam}))

if "--deprecated_fused_lamb" in sys.argv or "--cuda_ext" in sys.argv:
    if "--deprecated_fused_lamb" in sys.argv:
        sys.argv.remove("--deprecated_fused_lamb")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--deprecated_fused_lamb was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        print ("INFO: Building deprecated fused lamb extension.")
        nvcc_args_fused_lamb = ['-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_fused_lamb = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='fused_lamb_cuda',
                          sources=['apex/contrib/csrc/optimizers/fused_lamb_cuda.cpp',
                                   'apex/contrib/csrc/optimizers/fused_lamb_cuda_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args = {'nvcc' : nvcc_args_fused_lamb if not IS_ROCM_PYTORCH else hipcc_args_fused_lamb}))

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

if "--fast_layer_norm" in sys.argv:
    sys.argv.remove("--fast_layer_norm")
    # raise_if_cuda_home_none("--fast_layer_norm")
    # Check, if CUDA11 is installed for compute capability 8.0
    # cc_flag = []
    # _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
    # if int(bare_metal_major) >= 11:
    #     cc_flag.append("-gencode")
    #     cc_flag.append("arch=compute_80,code=sm_80")

    if CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--fast_layer_norm was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        if CUDA_HOME is not None:
            # Check, if CUDA11 is installed for compute capability 8.0
            cc_flag = []
            _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
            if int(bare_metal_major) >= 11:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_80,code=sm_80')
        else:
            hipcc_args_fused_lamb = ['-O3'] + version_dependent_macros
            ext_modules.append(
                CUDAExtension(
                    name="fast_layer_norm",
                    sources=[
                        "apex/contrib/csrc/layer_norm/ln_api.cpp",
                        "apex/contrib/csrc/layer_norm/ln_fwd_cuda_kernel.cu",
                        "apex/contrib/csrc/layer_norm/ln_bwd_semi_cuda_kernel.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                        "nvcc": [
                            "-O3",
                            '-U__HIP_NO_HALF_OPERATORS__',
                            '-U__HIP_NO_HALF_CONVERSIONS__',
                            "-I./apex/contrib/csrc/layer_norm/",
                        ] + version_dependent_macros + generator_flag,
                    },
                    include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/layer_norm")],
                )
            )

if "--fast_multihead_attn" in sys.argv or "--cuda_ext" in sys.argv:
    if "--fast_multihead_attn" in sys.argv:
        sys.argv.remove("--fast_multihead_attn")

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--fast_multihead_attn was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        # Check, if CUDA11 is installed for compute capability 8.0
        cc_flag = []
        if not IS_ROCM_PYTORCH:
            _, bare_metal_major, _ = get_cuda_bare_metal_version(torch.utils.cpp_extension.CUDA_HOME)
            if int(bare_metal_major) >= 11:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_80,code=sm_80')
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_86,code=sm_86')

        #subprocess.run(["git", "submodule", "update", "--init", "apex/contrib/csrc/multihead_attn/cutlass"])
        nvcc_args_mha = ['-O3',
                         '-gencode',
                         'arch=compute_70,code=sm_70',
                         '-Iapex/contrib/csrc/multihead_attn/cutlass',
                         '-U__CUDA_NO_HALF_OPERATORS__',
                         '-U__CUDA_NO_HALF_CONVERSIONS__',
                         '--expt-relaxed-constexpr',
                         '--expt-extended-lambda',
                         '--use_fast_math'] + version_dependent_macros + generator_flag + cc_flag
        hipcc_args_mha = ['-O3',
                          '-Iapex/contrib/csrc/multihead_attn/cutlass',
                          '-I' + os.path.join(ROCM_HOME, 'include/hiprand'),
                          '-I' + os.path.join(ROCM_HOME, 'include/rocrand'),
                          '-U__HIP_NO_HALF_OPERATORS__',
                          '-U__HIP_NO_HALF_CONVERSIONS__'] + version_dependent_macros + generator_flag
        if found_Backward_Pass_Guard:
            hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=BackwardPassGuard']
        if found_ROCmBackward_Pass_Guard:
            hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=ROCmBackwardPassGuard']

        ext_modules.append(
            CUDAExtension(
                name='fast_multihead_attn',
                sources=[
                    'apex/contrib/csrc/multihead_attn/multihead_attn_frontend.cpp',
                    'apex/contrib/csrc/multihead_attn/additive_masked_softmax_dropout_cuda.cu',
                    "apex/contrib/csrc/multihead_attn/masked_softmax_dropout_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/encdec_multihead_attn_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/encdec_multihead_attn_norm_add_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/self_multihead_attn_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/self_multihead_attn_bias_additive_mask_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/self_multihead_attn_bias_cuda.cu",
                    "apex/contrib/csrc/multihead_attn/self_multihead_attn_norm_add_cuda.cu",
                ],
                include_dirs=[os.path.join(this_dir, 'csrc'),
                                        os.path.join(this_dir, 'apex/contrib/csrc/multihead_attn')],
                          extra_compile_args={'cxx': ['-O3',] + version_dependent_macros + generator_flag,
                                              'nvcc':nvcc_args_mha if not IS_ROCM_PYTORCH else hipcc_args_mha}
            )
        )

if "--transducer" in sys.argv or "--cuda_ext" in sys.argv:
    if "--transducer" in sys.argv:
        sys.argv.remove("--transducer")
    
    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--transducer")

    hipcc_args_mha = ['-O3',
                        '-I' + os.path.join(ROCM_HOME, 'include/hiprand'),
                        '-I' + os.path.join(ROCM_HOME, 'include/rocrand'),] + version_dependent_macros + generator_flag
    ext_modules.append(
        CUDAExtension(
            name="transducer_joint_cuda",
            sources=[
                "apex/contrib/csrc/transducer/transducer_joint.cpp",
                "apex/contrib/csrc/transducer/transducer_joint_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": append_nvcc_threads(["-O3"] + version_dependent_macros + generator_flag) if not IS_ROCM_PYTORCH
                        else hipcc_args_mha,
            },
            include_dirs=[os.path.join(this_dir, "csrc"), os.path.join(this_dir, "apex/contrib/csrc/multihead_attn")],
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="transducer_loss_cuda",
            sources=[
                "apex/contrib/csrc/transducer/transducer_loss.cpp",
                "apex/contrib/csrc/transducer/transducer_loss_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": append_nvcc_threads(["-O3"] + version_dependent_macros) if not IS_ROCM_PYTORCH
                        else ["-O3"] + version_dependent_macros,
            },
        )
    )

if "--peer_memory" in sys.argv or "--cuda_ext" in sys.argv:
    if "--peer_memory" in sys.argv:
        sys.argv.remove("--peer_memory")

    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--peer_memory")

    ext_modules.append(
        CUDAExtension(
            name="peer_memory_cuda",
            sources=[
                "apex/contrib/csrc/peer_memory/peer_memory_cuda.cu",
                "apex/contrib/csrc/peer_memory/peer_memory.cpp",
            ],
            include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/nccl_p2p")],
            extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
        )
    )

if "--nccl_p2p" in sys.argv or "--cuda_ext" in sys.argv:
    if "--nccl_p2p" in sys.argv:
        sys.argv.remove("--nccl_p2p")

    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--nccl_p2p")

    ext_modules.append(
        CUDAExtension(
            name="nccl_p2p_cuda",
            sources=[
                "apex/contrib/csrc/nccl_p2p/nccl_p2p_cuda.cu",
                "apex/contrib/csrc/nccl_p2p/nccl_p2p.cpp",
            ],
            include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/nccl_p2p")],
            extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
        )
    )

if "--permutation_search" in sys.argv or "--cuda_ext" in sys.argv:
    if "--permutation_search" in sys.argv:
        sys.argv.remove("--permutation_search")

    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--permutation_search")

    cc_flag = ["-Xcompiler", "-fPIC", "-shared"]
    ext_modules.append(
        CUDAExtension(
            name="permutation_search_cuda",
            sources=[
                "apex/contrib/sparsity/permutation_search_kernels/CUDA_kernels/permutation_search_kernels.cu"
            ],
            include_dirs=[
                os.path.join(
                    this_dir,
                    "apex",
                    "contrib",
                    "sparsity",
                    "permutation_search_kernels",
                    "CUDA_kernels",
                )
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + cc_flag},
        )
    )

if "--group_norm" in sys.argv or "--cuda_ext" in sys.argv:
    if "--group_norm" in sys.argv:
        sys.argv.remove("--group_norm")

    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--group_norm")

    ext_modules.append(
        CUDAExtension(
            name="group_norm_cuda",
            sources=[
                "apex/contrib/csrc/group_norm/group_norm_nhwc_op.cpp",
            ]
            + glob.glob("apex/contrib/csrc/group_norm/*.cu"),
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3", "--use_fast_math"] + version_dependent_macros + generator_flag,
            },
        )
    )

if "--nccl_allocator" in sys.argv or "--cuda_ext" in sys.argv:
    if "--nccl_allocator" in sys.argv:
        sys.argv.remove("--nccl_allocator")
    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--nccl_allocator")
    _nccl_version_getter = load(
        name="_nccl_version_getter",
        sources=[
            "apex/contrib/csrc/nccl_p2p/nccl_version.cpp",
            "apex/contrib/csrc/nccl_p2p/nccl_version_check.cu",
        ],
    )
    _available_nccl_version = _nccl_version_getter.get_nccl_version()
    if _available_nccl_version >= (2, 19):
        ext_modules.append(
            CUDAExtension(
                name="_apex_nccl_allocator",
                sources=[
                    "apex/contrib/csrc/nccl_allocator/NCCLAllocator.cpp",
                ],
                include_dirs=[os.path.join(this_dir, "apex/apex/contrib/csrc/nccl_allocator")],
                libraries=["rccl" if IS_ROCM_PYTORCH else "nccl"],
                extra_compile_args={"cxx": ["-O3"] + generator_flag},
            )
        )
    else:
        warnings.warn(
            f"Skip `--nccl_allocator` as it requires NCCL 2.19 or later, but {_available_nccl_version[0]}.{_available_nccl_version[1]}"
        )

if "--gpu_direct_storage" in sys.argv or "--cuda_ext" in sys.argv:
    if "--gpu_direct_storage" in sys.argv:
        sys.argv.remove("--gpu_direct_storage")
    if not IS_ROCM_PYTORCH:
        raise_if_cuda_home_none("--gpu_direct_storage")
    hipfile_available = IS_ROCM_PYTORCH and os.path.exists(os.path.join(ROCM_HOME, "include", "hipfile.h"))
    if IS_ROCM_PYTORCH and not hipfile_available:
        warnings.warn("hipFile is unavailable; GPU Direct Storage will use the POSIX I/O fallback")
    ext_modules.append(
        CUDAExtension(
            name="_apex_gpu_direct_storage",
            sources=[
                "apex/contrib/csrc/gpu_direct_storage/gds.cpp",
                "apex/contrib/csrc/gpu_direct_storage/gds_pybind.cpp",
            ],
            include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/gpu_direct_storage")],
            libraries=["hipfile"] if hipfile_available else ([] if IS_ROCM_PYTORCH else ["cufile"]),
            extra_compile_args={"cxx": ["-O3"] + generator_flag},
        )
    )
    
if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")

setup(
    name="apex",
    version=hcu_version,
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    description="PyTorch Apex extensions adapted for DAS/HCU and DTK",
    license="BSD-3-Clause",
    url="https://github.com/HYGON-AI/apex-das",
    project_urls={
        "Upstream": "https://github.com/ROCm/apex",
        "Source": "https://github.com/HYGON-AI/apex-das",
    },
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    extras_require=extras,
)
