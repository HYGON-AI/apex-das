# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.resolve()


def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None


def _get_version():
    if os.path.exists(ROOT_DIR / "version.txt"):
        with open(ROOT_DIR / "version.txt", "r") as f:
            version = f.read().strip()
    else:
        version = '0.1'
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    return version


def _make_version_file(version, das_version, sha, abi, dtk, torch_version, branch):
    sha = "Unknown" if sha is None else sha
    # torch_version = '.'.join(torch_version.split('.')[:2])
    # hcu_version = f"{das_version}.git{sha}.abi{abi}.dtk{dtk}.torch{torch_version}"
    hcu_version = f"{das_version}"
    version_path = ROOT_DIR / "apex" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"version = '{version}'\n")
        f.write(f"git_hash = '{sha}'\n")
        f.write(f"git_branch = '{branch}'\n")
        f.write(f"abi = 'abi{abi}'\n")
        f.write(f"dtk = '{dtk}'\n")
        f.write(f"torch_version = '{torch_version}'\n")
        f.write(f"hcu_version = '{hcu_version}'\n")
    return hcu_version


def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

def get_version(ROCM_HOME):
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    sha = sha[:7] if sha else "Unknown"
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "Unknown"
    tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])
    print("-- Git branch:", branch)
    print("-- Git SHA:", sha)
    print("-- Git tag:", tag)
    torch_version = _get_pytorch_version()
    print("-- PyTorch:", torch_version)
    version = _get_version()
    print("-- Building version", version)
    optimization_version = os.getenv("DAS_OPT_VERSION", "opt2")
    das_version = f"{version}+das.{optimization_version}"
    print("-- Building das_version", das_version)
    abi = int(torch.compiled_with_cxx11_abi())
    print("-- _GLIBCXX_USE_CXX11_ABI:", abi)
    dtk_version = os.getenv("DTK_VERSION")
    if not dtk_version and ROCM_HOME:
        version_file = Path(ROCM_HOME) / ".info" / "rocm_version"
        if version_file.exists():
            dtk_version = version_file.read_text(encoding="utf-8").strip()
    if not dtk_version:
        dtk_version = torch.version.hip or "unknown"
    dtk = "".join(dtk_version.split("."))
    print("-- DTK:", dtk)
    das_version += ".dtk" + dtk

    return _make_version_file(version, das_version, sha, abi, dtk, torch_version, branch)
