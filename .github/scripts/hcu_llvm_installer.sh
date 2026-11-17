#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# hcu aicc 编译器安装脚本
#
# 用法:
#   ./{script}                                             # 默认装最新版本(版本号最高, 该版本内日期最新)
#   ./{script} --major 2.0.0                               # 指定大版本号, 装该版本中日期最新的包
#   ./{script} --version 2.0.0+dtk26042.2609071657.ga995da # 指定完整版本, 精确匹配该包
#   ./{script} --commit g16beee                            # 指定 commit, 装对应包(可省略前缀 g, 支持短 hash 前缀匹配)
#   ./{script} --dtk-dir /opt/dtk                          # 指定 dtk 安装目录, 默认 /opt/dtk
#   ./{script} --dry-run                                   # 只打印将要下载安装的包, 不实际下载安装
#   ./{script} --keep                                      # 安装成功后保留下载的 run 包(默认装完自动删除)
#
# 安装方式: yes | bash <包> --dtk_dir <目录>

set -euo pipefail

BASE_URL="http://42.228.13.241:18000/ai_cc/Nightly/"
PRODUCT="dtk_llvm"          # 匹配该产品名; 2026-09 起目录里的包改名带 _aicc 后缀, 新旧名都收
ARCH="x86_64"               # 只匹配该架构的包, 置空则不过滤架构
DTK_DIR="/opt/dtk"
DRY_RUN=0
KEEP=0

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' | sed "s|{script}|$(basename "$0")|g"
    exit "${1:-0}"
}

MAJOR=""
VERSION=""
COMMIT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --major)     MAJOR="$2"; shift 2 ;;
        --version)   VERSION="$2"; shift 2 ;;
        --commit)    COMMIT="$2"; shift 2 ;;
        --dtk-dir)   DTK_DIR="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --keep)      KEEP=1; shift ;;
        -h|--help)   usage ;;
        *) echo "未知参数: $1"; usage 1 ;;
    esac
done

n_sel=0
[[ -n "$MAJOR" ]] && n_sel=$((n_sel + 1))
[[ -n "$VERSION" ]] && n_sel=$((n_sel + 1))
[[ -n "$COMMIT" ]] && n_sel=$((n_sel + 1))
[[ "$n_sel" -gt 1 ]] && { echo "--major / --version / --commit 只能指定一个"; exit 1; }

# dry-run 不接触磁盘, 不做目录检查
if [[ "$DRY_RUN" -eq 0 && ! -d "$DTK_DIR" ]]; then
    echo "错误: 安装目录 $DTK_DIR 不存在, 请先创建(需要时用 sudo): sudo mkdir -p $DTK_DIR"
    exit 1
fi

# 拉取目录索引, 解析所有 run 包
# 文件名格式: {product}-{ver}+{branch}.{yymmddHHMM}.g{shortcommit}-{arch}.run
index=$(curl -fsSL --max-time 30 "$BASE_URL") || { echo "无法访问 $BASE_URL"; exit 1; }
parse_re='^([A-Za-z0-9_]+)-([^+]+)\+([^.]*)\.([0-9]{10})\.(g[0-9a-f]+)-([A-Za-z0-9_]+)\.run$'

candidates=()
while IFS= read -r href; do
    [[ -z "$href" ]] && continue
    fname="${href//%2B/+}"
    [[ "$fname" =~ $parse_re ]] || continue
    name="${BASH_REMATCH[1]}"
    ver="${BASH_REMATCH[2]}"
    br="${BASH_REMATCH[3]}"
    ts="${BASH_REMATCH[4]}"
    cmt="${BASH_REMATCH[5]}"
    arch="${BASH_REMATCH[6]}"
    # 产品名兼容: dtk_llvm 和改名后的 dtk_llvm_aicc 视为同一种包, 都收进候选
    [[ "$name" == "$PRODUCT" || "$name" == "${PRODUCT}_aicc" ]] || continue
    [[ -n "$ARCH" && "$arch" != "$ARCH" ]] && continue
    full="${ver}+${br}.${ts}.${cmt}"
    candidates+=("$ts $ver $full $cmt $fname")
done < <(printf '%s' "$index" | sed -n 's/.*href="\([^"]*\.run\)".*/\1/p')

[[ ${#candidates[@]} -eq 0 ]] && { echo "目录中没有找到 $PRODUCT 的 $ARCH 包: $BASE_URL"; exit 1; }

# candidates 统一按日期时间戳降序排
printf '%s\n' "${candidates[@]}" | sort -k1,1r > /tmp/dtk_candidates.$$

pick=""
if [[ -n "$COMMIT" ]]; then
    c="${COMMIT,,}"
    c="${c#g}"                                   # 去掉可选的 g 前缀
    [[ -z "$c" ]] && { echo "commit 不能为空"; exit 1; }
    pick=$(awk -v pref="g$c" 'substr($4, 1, length(pref)) == pref { print; exit }' /tmp/dtk_candidates.$$)
    if [[ -z "$pick" ]]; then
        echo "没有找到 commit=$COMMIT 的包, 当前可用的 commit:"
        awk '{print $4}' /tmp/dtk_candidates.$$ | sort -u
        exit 1
    fi
elif [[ -n "$VERSION" ]]; then
    pick=$(awk -v v="$VERSION" '$3 == v { print; exit }' /tmp/dtk_candidates.$$)
    if [[ -z "$pick" ]]; then
        echo "没有找到完整版本 $VERSION 的包, 当前可用的完整版本(形式如 2.0.0+dtk26042.2609071657.ga995da):"
        awk '{print $3}' /tmp/dtk_candidates.$$ | sort -u
        exit 1
    fi
elif [[ -n "$MAJOR" ]]; then
    pick=$(awk -v m="$MAJOR" '$2 == m { print; exit }' /tmp/dtk_candidates.$$)   # 已按日期降序, 取第一个=日期最新
    if [[ -z "$pick" ]]; then
        echo "没有找到大版本 $MAJOR 的包, 当前可用的大版本:"
        awk '{print $2}' /tmp/dtk_candidates.$$ | sort -Vu
        exit 1
    fi
else
    # 默认: 版本号最高, 该版本内取日期最新
    pick=$(sort -k2,2Vr -k1,1r /tmp/dtk_candidates.$$ | head -1)
fi
rm -f /tmp/dtk_candidates.$$

read -r ts ver full cmt fname <<<"$pick"
url="${BASE_URL}${fname//+/%2B}"

echo "==== 选定的包 ===="
echo "完整版本: $full"
echo "日期:     20${ts:0:2}-${ts:2:2}-${ts:4:2} ${ts:6:2}:${ts:8:2}"
echo "commit:   $cmt"
echo "文件:     $fname"
echo "安装目录: $DTK_DIR"
echo "下载:     $url"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "==== dry-run 结束, 未下载未安装 ===="
    exit 0
fi

if [[ ! -f "$fname" ]]; then
    echo "==== 开始下载 ===="
    curl -fSL --retry 3 --retry-delay 2 -o "$fname" "$url"
fi

echo "==== 开始安装 ===="
# 安装器退出后 yes 会收到 SIGPIPE(141), 开着 pipefail 会把成功安装误判为失败,
# 导致脚本提前退出、清理代码不执行; 子 shell 里关掉 pipefail, 保留安装器真实退出码
(
    set +o pipefail
    yes | bash "$fname" --dtk_dir "$DTK_DIR"
)

echo "==== 安装完成: $fname -> $DTK_DIR ===="

# 在安装目录留下本次安装的版本信息, 覆盖旧记录, 始终反映当前生效的版本
if ! cat > "$DTK_DIR/hcu_llvm.info" <<EOF
==== hcu aicc 编译器安装记录 (每次安装覆盖) ====
安装时间:  $(date '+%Y-%m-%d %H:%M:%S')
安装目录:  $DTK_DIR
完整版本:  $full
commit:    $cmt
构建时间:  20${ts:0:2}-${ts:2:2}-${ts:4:2} ${ts:6:2}:${ts:8:2}
安装包:    $fname
EOF
then
    echo "警告: 写入 $DTK_DIR/hcu_llvm.info 失败(不影响安装结果)"
fi

# 装完即删(除非 --keep)。失败/中断时走不到这里, 不会误删
if [[ "$KEEP" -eq 0 ]]; then
    rm -f "$fname"
    echo "==== 已删除安装包: $fname (加 --keep 可保留) ===="
fi
