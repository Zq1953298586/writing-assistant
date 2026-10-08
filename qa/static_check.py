#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态检查（用法: python3 static_check.py <导出的html路径>）：
1. 所有内联 <script> 用 node --check 做语法检查（无 node 则跳过并警告）
2. 基本结构（html/head/body）
3. 外部网络依赖（src/href 指向公网）→ 警告
4. viewport meta（手机适配）→ 缺失则警告
FAIL 只判：JS 语法错误、缺基本结构。其余为 WARN。
退出码：0=通过（含警告），1=失败。"""
import re
import shutil
import subprocess
import sys
import tempfile
import os

path = sys.argv[1]
fails, warns = [], []

html = open(path, encoding="utf-8", errors="replace").read()
print(f"文件: {path}")
print(f"大小: {len(html) // 1024} KB")

# 1. 基本结构
for tag in ("<html", "<head", "<body"):
    if tag not in html.lower():
        fails.append(f"缺基本结构: {tag}")

# 2. JS 语法检查
scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html,
                     re.S | re.I)
print(f"内联 script 块: {len(scripts)}")
node = shutil.which("node")
if not node:
    warns.append("无 node，跳过 JS 语法检查")
else:
    for i, code in enumerate(scripts):
        if not code.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            r = subprocess.run([node, "--check", tmp],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                fails.append(f"script[{i}] 语法错误: "
                             f"{(r.stderr or r.stdout).strip()[:300]}")
        finally:
            os.unlink(tmp)

# 3. 外部网络依赖
ext = re.findall(r'''(?:src|href)\s*=\s*["'](https?://[^"']+)["']''',
                 html, re.I)
ext = [u for u in ext if "localhost" not in u and "127.0.0.1" not in u]
if ext:
    warns.append(f"发现 {len(ext)} 个外部网络依赖（示例: {ext[0]}），"
                 "离线/弱网下可能失效")

# 4. viewport
if "viewport" not in html.lower():
    warns.append("缺 viewport meta，手机显示可能有问题")

# 5. 功能关键词抽查（仅提示，不判 FAIL，以浏览器实测为准）
markers = ["续写", "润色", "仿写", "批评", "风格", "上传", "复制",
           "localStorage"]
missing = [m for m in markers if m not in html]
if missing:
    warns.append(f"未找到关键词 {missing}，浏览器实测时重点验证对应功能")

print("----")
for w in warns:
    print("WARN:", w)
for f in fails:
    print("FAIL:", f)
if fails:
    print(f"结果: FAIL（{len(fails)} 项）")
    sys.exit(1)
print(f"结果: PASS（{len(warns)} 条警告）")
