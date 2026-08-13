#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# -*- coding: utf-8 -*-
"""将 Apex 单测日志(apex_test.log)解析为 JUnit XML, 用于 GitLab CI 的 Test Report 界面。

仅使用 Python 标准库, 无第三方依赖。解析规则与
backend/app/routers/files.py 中处理 apex_test.log 的 unittest 解析逻辑
(_parse_unittest_summary / _extract_unittest_details / _extract_unittest_failed_details)
保持一致, 便于与网页端 "测试摘要" 的结果对齐。

用法:
    python parse_apex_log_to_junit.py [apex_test.log] [-o junit.xml] [--suite-name apex_test]

示例 (GitLab CI, .gitlab-ci.yml):
    run_apex_test:
      stage: test
      script:
        - bash run_rocm.sh > apex_test.log 2>&1 || true   # 跑完测试, 即使失败也保留日志
        - python parse_apex_log_to_junit.py apex_test.log -o apex_test_junit.xml
      artifacts:
        when: always
        reports:
          junit: apex_test_junit.xml
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

# ---- 解析用正则 (与 backend/app/routers/files.py 对齐) ----

# 去掉 ANSI 转义序列 (日志中常见颜色控制码)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# "test_groupbn (unittest.loader._FailedTest) ... ERROR" 这类单行结果
_UNITTEST_DETAIL_RE = re.compile(
    r"(?P<name>.+?)\s+\.\.\.\s+(?P<status>ok|FAIL|ERROR|skipped.*)$", re.IGNORECASE
)

# "=====\nERROR: <name>\n-----\n<traceback>..." 错误块
_ERROR_BLOCK_RE = re.compile(
    r"^={20,}\n(?P<kind>FAIL|ERROR):[ \t]*(?P<name>[^\n]+)\n"
    r"(?P<description>.*?)^-{20,}\n(?P<body>.*?)"
    r"(?=^={20,}|\n-{20,}\nRan\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)

# 收集失败用例 (unittest 汇总行, 用于兜底)
_FAILED_LOADER_RE = re.compile(
    r"^(\S+(?:\.\S+)*)\s+\(unittest\.loader\._FailedTest\)\s+\.\.\.\s+ERROR$", re.MULTILINE
)

# 每个 unittest 子块的 "Ran N tests in ..." 汇总
_RAN_RE = re.compile(r"^Ran\s+(\d+)\s+tests?\s+in\s+", re.MULTILINE)

# "FAILED (errors=2, failures=1)" / "OK (skipped=1)"
_FAILED_SUMMARY_RE = re.compile(r"^FAILED\s+\(([^)]*)\)", re.MULTILINE)
_OK_SUMMARY_RE = re.compile(r"^OK\s*(?:\(([^)]*)\))?", re.MULTILINE)

# traceback 里的源文件路径, 用于推断分组 (classname)
_FILE_RE = re.compile(r'File "(?P<path>[^"]+\.py)"')

# 提取首条失败原因 (与 backend _first_failure_reason 对齐)
_FAILURE_PATTERNS = (
    r"ModuleNotFoundError: .*",
    r"ImportError: .*",
    r"FileNotFoundError: .*",
    r"AttributeError: .*",
    r"AssertionError: .*",
    r"RuntimeError: .*",
    r"TypeError: .*",
    r"OSError: .*",
    r"ERROR: .*",
    r"Error: .*",
    r".*\bFAILED\b.*",
    r".*\bFAIL\b.*",
    r"No such file or directory.*",
)

_TRACEBACK_MAX = 4000  # 单个用例 failure/error 节点正文的最大长度


class TestResult:
    """一个测试用例的解析结果。"""

    def __init__(self, name: str, status: str, kind: str, reason: str, body: str, classname: str):
        self.name = name        # 展示名, 如 test_groupbn
        self.status = status    # passed / failed / skipped
        self.kind = kind        # FAIL / ERROR / "" (passed/skipped 时为空)
        self.reason = reason    # 一句话原因
        self.body = body        # 完整 traceback (已截断)
        self.classname = classname or "apex"


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    text = _ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))
    return text


def clean_name(raw: str) -> str:
    """去掉 unittest.loader._FailedTest 后缀, 得到展示名。"""
    return raw.replace(" (unittest.loader._FailedTest)", "").strip()


def map_status(raw_status: str) -> str:
    """把行尾状态词映射为 passed / skipped / failed (与 backend _detail_status 对齐)。"""
    s = raw_status.lower()
    if s in ("ok", "passed", "pass", "xpass") or s.startswith("subpassed"):
        return "passed"
    if s.startswith("skip") or s == "xfail":
        return "skipped"
    return "failed"


def failure_reason(body: str) -> str:
    """从错误块底部向上找第一条真实原因 (与 backend _failure_reason 对齐)。"""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("File ") or line.startswith("^"):
            continue
        return line
    return ""


def first_failure_reason(text: str) -> str:
    """全文搜第一条失败原因 (与 backend _first_failure_reason 对齐)。"""
    for pattern in _FAILURE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def infer_classname(body: str, fallback_name: str) -> str:
    """从 traceback 源文件路径推断分组, 例如 .../test/groupbn/test_groupbn.py -> groupbn。

    只有测试文件位于某个子目录下才返回该目录; 直接放在 test/ 下的用例返回空串。
    """
    for m in _FILE_RE.finditer(body):
        parts = [p for p in m.group("path").replace("\\", "/").split("/") if p]
        try:
            idx = parts.index("test")
        except ValueError:
            continue
        rest = parts[idx + 1 :]
        if len(rest) > 1:
            return "/".join(rest[:-1])
    return ""


def error_type(reason: str) -> str:
    m = re.match(r"([A-Za-z_]\w*(?:Error|Exception|Warning))", reason)
    return m.group(1) if m else "unittest.Error"


def parse_log(text: str) -> tuple[list[TestResult], dict[str, int]]:
    """解析日志, 返回用例列表与汇总计数。"""
    results: list[TestResult] = []
    seen: set[tuple[str, str]] = set()

    # 1) 收集每条 "name ... ok/FAIL/ERROR/skipped" 结果行
    for line in text.splitlines():
        m = _UNITTEST_DETAIL_RE.match(line.strip())
        if not m:
            continue
        raw_status = m.group("status")
        status = map_status(raw_status)
        name = clean_name(m.group("name"))
        key = (name, status)
        if key in seen:
            continue
        seen.add(key)
        kind = raw_status.upper() if raw_status.upper() in ("FAIL", "ERROR") else ""
        results.append(TestResult(name, status, kind, "", "", ""))

    # 2) 用错误块填充失败用例的原因与 traceback, 并推断分组。
    # 默认 verbosity 和 subTest 失败可能只有 FAIL:/ERROR: 块，没有明细结果行。
    blocks: dict[str, tuple[str, str]] = {}  # 干净名 -> (FAIL/ERROR, body)
    for m in _ERROR_BLOCK_RE.finditer(text):
        key = clean_name(m.group("name"))
        if key not in blocks:
            description = m.group("description").strip()
            body = m.group("body")
            if description:
                body = f"{description}\n{body}"
            blocks[key] = (m.group("kind").upper(), body)

    for r in results:
        if r.status != "failed":
            continue
        block = blocks.get(r.name)
        body = block[1] if block else ""
        if block:
            r.kind = block[0]
        reason = first_failure_reason(body) or failure_reason(body) or first_failure_reason(text)
        r.reason = reason or "unittest 报告该用例失败, 日志中未记录具体异常"
        r.body = body.strip()[:_TRACEBACK_MAX]
        r.classname = infer_classname(body, r.name) or r.classname

    failed_names = {r.name for r in results if r.status == "failed"}
    for name, (kind, body) in blocks.items():
        if name in failed_names:
            continue
        reason = first_failure_reason(body) or failure_reason(body)
        results.append(TestResult(
            name=name,
            status="failed",
            kind=kind,
            reason=reason or "unittest 报告该用例失败, 日志中未记录具体异常",
            body=body.strip()[:_TRACEBACK_MAX],
            classname=infer_classname(body, name) or "apex",
        ))
        failed_names.add(name)

    # 3) 兜底: 只有汇总行、没有完整结果行的失败用例
    for m in _FAILED_LOADER_RE.finditer(text):
        name = clean_name(m.group(1))
        if not any(r.name == name and r.status == "failed" for r in results):
            reason = first_failure_reason(text) or "unittest 报告该用例失败, 日志中未记录具体异常"
            results.append(TestResult(name, "failed", "ERROR", reason, "", ""))

    # 4) 汇总计数: 与 backend _parse_unittest_summary 保持一致
    total = sum(int(n) for n in _RAN_RE.findall(text))
    failures = 0
    errors = 0
    skipped = 0
    for body in _FAILED_SUMMARY_RE.findall(text):
        m = re.search(r"failures=(\d+)", body)
        if m:
            failures += int(m.group(1))
        m = re.search(r"errors=(\d+)", body)
        if m:
            errors += int(m.group(1))
        m = re.search(r"skipped=(\d+)", body)
        if m:
            skipped += int(m.group(1))
    for body in _OK_SUMMARY_RE.findall(text):
        if not body:
            continue
        m = re.search(r"skipped=(\d+)", body)
        if m:
            skipped += int(m.group(1))
    parsed_failures = sum(r.status == "failed" and r.kind != "ERROR" for r in results)
    parsed_errors = sum(r.status == "failed" and r.kind == "ERROR" for r in results)
    parsed_skipped = sum(r.status == "skipped" for r in results)
    parsed_passed = sum(r.status == "passed" for r in results)

    # 汇总行有失败但详细日志被截断时，也要让 Reporter 显示明确的失败用例。
    for index in range(parsed_failures, failures):
        results.append(TestResult(
            name=f"unreported unittest failure #{index + 1}",
            status="failed",
            kind="FAIL",
            reason="unittest summary reported a failure without a matching detail block",
            body="",
            classname="apex.unreported",
        ))
    for index in range(parsed_errors, errors):
        results.append(TestResult(
            name=f"unreported unittest error #{index + 1}",
            status="failed",
            kind="ERROR",
            reason="unittest summary reported an error without a matching detail block",
            body="",
            classname="apex.unreported",
        ))

    failures = max(failures, sum(r.status == "failed" and r.kind != "ERROR" for r in results))
    errors = max(errors, sum(r.status == "failed" and r.kind == "ERROR" for r in results))
    skipped = max(skipped, parsed_skipped)
    failed = failures + errors
    total = max(total, len(results), parsed_passed + failed + skipped)
    passed = max(total - failed - skipped, parsed_passed, 0)

    return results, {
        "total": total,
        "passed": passed,
        "failed": failed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def build_junit(results: list[TestResult], counts: dict[str, int], suite_name: str) -> str:
    """组装 GitLab 兼容的 JUnit XML。"""

    errors = max(counts.get("errors", 0), sum(1 for r in results if r.kind == "ERROR"))
    failures = max(counts.get("failures", 0), sum(1 for r in results if r.kind == "FAIL"))
    skipped = max(counts["skipped"], sum(1 for r in results if r.status == "skipped"))
    # counts 来自 Ran/FAILED 汇总行; 若用例行更多, 以实际为准, 保证不丢用例
    total = max(counts["total"], len(results))

    suite_el = ET.Element("testsuite", {
        "name": suite_name,
        "tests": str(total),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
    })

    for r in results:
        tc = ET.SubElement(suite_el, "testcase", {
            "classname": r.classname,
            "name": r.name,
            "time": "0",
        })
        if r.status == "skipped":
            ET.SubElement(tc, "skipped")
        elif r.status == "failed":
            detail = r.body or r.reason
            attrib = {"message": r.reason, "type": error_type(r.reason)}
            if r.kind == "ERROR":
                ET.SubElement(tc, "error", attrib).text = detail
            else:
                ET.SubElement(tc, "failure", attrib).text = detail

    suites_el = ET.Element("testsuites", {
        "name": suite_name,
        "tests": str(total),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
    })
    suites_el.append(suite_el)

    rough = ET.tostring(suites_el, encoding="unicode")
    pretty = minidom.parseString(ET.tostring(suites_el, encoding="utf-8"))
    return pretty.toprettyxml(indent="  ", encoding=None)


def main() -> int:
    ap = argparse.ArgumentParser(description="解析 apex_test.log 并输出 JUnit XML (GitLab CI 测试报告)")
    ap.add_argument("log", nargs="?", default="apex_test.log", help="apex_test.log 路径 (默认: ./apex_test.log)")
    ap.add_argument("-o", "--output", default="apex_test_junit.xml", help="输出 JUnit XML 路径 (默认: ./apex_test_junit.xml)")
    ap.add_argument("--suite-name", default="apex_test", help="testsuite 名称 (默认: apex_test)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"错误: 日志文件不存在: {log_path}", file=sys.stderr)
        return 2

    text = read_log(log_path)
    if not text.strip():
        print(f"错误: 日志文件为空: {log_path}", file=sys.stderr)
        return 2

    results, counts = parse_log(text)
    xml_text = build_junit(results, counts, args.suite_name)

    out = Path(args.output)
    out.write_text(xml_text, encoding="utf-8")

    print(f"解析完成: {log_path}")
    print(f"  total={counts['total']}  passed={counts['passed']}  failed={counts['failed']}  skipped={counts['skipped']}")
    print(f"  (统计到 {len(results)} 个用例详情)")
    print(f"JUnit XML 已写入: {out}")

    if counts["failed"]:
        print(f"失败原因(前几个):")
        for r in results:
            if r.status == "failed":
                print(f"  - [{r.classname}] {r.name}: {r.reason}")
                if len(results) >= 5:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
