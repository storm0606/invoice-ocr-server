#!/usr/bin/env python3
"""
下载 PP-OCRv6_medium ONNX 模型。

模型文件:
  - ch_PP-OCRv6_det_models.onnx  → models/det/inference.onnx  (检测)
  - ch_PP-OCRv6_rec_medium.onnx  → models/rec_medium/inference.onnx  (识别)

用法:
    python tools/download_models.py

注意: 模型合计约 25MB，需要联网。
"""

import os
import sys
import urllib.request
from pathlib import Path

# 添加项目根目录到 Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR, DET_MODEL_URL, REC_MODEL_URL


def download_file(url: str, dest: str, desc: str) -> None:
    """下载文件并显示进度"""
    print(f"📥 {desc}")

    def report(block, blocks, total):
        downloaded = block * blocks
        if total > 0:
            pct = min(100, downloaded * 100 // total)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  {bar} {pct}% ({downloaded // 1024}KB / {total // 1024}KB)", end="")
        else:
            print(f"\r  {downloaded // 1024}KB", end="")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=report)
    size = os.path.getsize(dest)
    print(f"\n  ✓ 完成 ({size / 1024 / 1024:.1f}MB)")
    print()


def main():
    print("=" * 50)
    print("PP-OCRv6_medium ONNX 模型下载")
    print("=" * 50)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 下载检测模型
    det_dest = os.path.join(MODELS_DIR, "det", "inference.onnx")
    if os.path.exists(det_dest):
        size = os.path.getsize(det_dest) / 1024 / 1024
        print(f"✓ 检测模型已存在: {det_dest} ({size:.1f}MB)")
    else:
        download_file(DET_MODEL_URL, det_dest, "检测模型 (ch_PP-OCRv6_det)")

    # 下载识别模型 (medium)
    rec_dest = os.path.join(MODELS_DIR, "rec_medium", "inference.onnx")
    if os.path.exists(rec_dest):
        size = os.path.getsize(rec_dest) / 1024 / 1024
        print(f"✓ 识别模型已存在: {rec_dest} ({size:.1f}MB)")
    else:
        download_file(REC_MODEL_URL, rec_dest, "识别模型 (ch_PP-OCRv6_rec_medium)")

    print("=" * 50)
    print("所有模型下载完成!")
    print(f"  检测模型: {det_dest}")
    print(f"  识别模型: {rec_dest}")
    print()
    print("启动服务：python server.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
