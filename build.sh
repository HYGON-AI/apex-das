#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_JOBS="${MAX_JOBS:-16}"
export MAX_JOBS

"${PYTHON_BIN}" -m pip install --upgrade setuptools wheel
CXX="${CXX:-hipcc}" CC="${CC:-hipcc}" \
  "${PYTHON_BIN}" setup.py --cpp_ext --cuda_ext bdist_wheel

echo "Wheel created under dist/. Install it with:"
echo "  ${PYTHON_BIN} -m pip install dist/apex-*.whl"
