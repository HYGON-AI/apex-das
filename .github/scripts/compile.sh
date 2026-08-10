#!/bin/bash
set -e
source /opt/dtk/env.sh
apex_version=$(cat version.txt)
torch_version=$(python3 -c "import torch; print(torch.__version__)")
new_version=`echo ${torch_version} | tr -d '.'`
dtk_version=$(cat /opt/dtk/.dtk_version |head -n 1| awk -F '-' '{print $2}' | tr -d '.')
new_version="hcu_version=\"${apex_version}+das.opt1.dtk${dtk_version}\""
line=$(grep -nr "setup(" setup.py | awk -F ":" '{print $1}')
sed -i "${line}i\\${new_version}" setup.py
export PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS}
export MAX_JOBS=16
export CFLAGS="-w"
export CXXFLAGS="-w"
CXX=hipcc CC=hipcc python3 setup.py --cpp_ext --cuda_ext  bdist_wheel