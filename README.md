# APEX for DAS

apex-das 是面向 DAS/HCU 平台和 DTK 软件栈适配的 PyTorch Apex 扩展库，提供混合精度训练、分布式训练和 HCU/HIP Kernel 融合优化能力。

本项目基于 [ROCm/apex](https://github.com/ROCm/apex) 二次开发；ROCm Apex 源自 [NVIDIA/apex](https://github.com/NVIDIA/apex)。

- 上游分支：`release/1.7.0`
- 上游 Commit：`86fff107d5e73043c75003c1530dbeabce8984fb`
- 上游许可证：`BSD-3-Clause`

Modified by Hygon Information Technology Co., Ltd.
本仓库包含由 Hygon Information Technology Co., Ltd. 完成的 DAS/HCU 平台适配与修改。

## 主要适配

- DTK/HIP 编译与 HCU 架构适配。
- Fused Dense、MLP、FusedAdam、AMP、GroupNorm、LayerNorm 和 Transducer Joint 兼容性优化。
- GPU Direct Storage：优先使用 hipFile，不可用时回退到 POSIX I/O。
- RCCL Allocator、稀疏训练和排列搜索扩展。
- PyTorch autocast、分布式启动参数及相关测试适配。


## 版本配套

| Apex | DAS 软件包版本 | HCU 型号 | DTK |
| --- | --- | --- | --- |
| 1.7.0 | apex-1.7.0+das.opt2.dtk2604 | Z100、Z100L、K100、K100_AI、BW系列 | 26.04 |

## 环境准备

请先安装 HCU 驱动、DTK、DAS PyTorch 和 Python 构建工具，并加载 DTK 环境：

~~~bash
source /opt/dtk/env.sh
export PYTORCH_ROCM_ARCH="gfx906;gfx926;gfx928;gfx936"
export MAX_JOBS=16
python3 -m pip install -r requirements.txt
python3 -m pip install wheel
~~~

## 构建与安装

~~~bash
CXX=hipcc CC=hipcc python3 setup.py --cpp_ext --cuda_ext bdist_wheel
python3 -m pip install dist/apex-*.whl
~~~

也可使用：

~~~bash
./build.sh
~~~

## 已知限制

- 仅支持表格中列出的 DTK/HCU 组合。
- Transducer 仅完成部分功能适配。
- GroupNorm 的新版上游代码已引入，暂不支持GroupNorm v2。

## License

仓库主许可证为 [BSD 3-Clause](LICENSE)。第三方组件继续适用其各自许可证和版权声明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [LICENSES](LICENSES/)。

Copyright (c) 2026 Hygon Information Technology Co., Ltd.
Hygon 的版权声明仅适用于其新增或修改内容，不替代 ROCm、NVIDIA 或其他上游作者的版权声明。
