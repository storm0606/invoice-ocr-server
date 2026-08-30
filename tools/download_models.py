#!/usr/bin/env python3
"""
下载 PP-OCRv6_medium ONNX 模型。

模型来源:
  https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium

PaddleX 在首次运行时会自动下载默认 OCR 模型。
如果需要使用 PP-OCRv6_medium，可以从 Hugging Face 下载。

用法:
    python tools/download_models.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Hugging Face 模型文件
HF_BASE = "https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium/resolve/main"

MODEL_FILES = {
    "det": {
        "url": f"{HF_BASE}/ch_PP-OCRv6_det/inference.onnx",
        "subdir": "ch_PP-OCRv6_det",
    },
    "rec": {
        "url": f"{HF_BASE}/ch_PP-OCRv6_rec_medium/inference.onnx",
        "subdir": "ch_PP-OCRv6_rec_medium",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_file(url: str, dest: Path, label: str):
    """Download a file with progress indicator"""
    print(f"\n📥 {label}")
    print(f"   从: {url}")
    print(f"   到: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    def report(block, blocksize, totalsize):
        if totalsize > 0:
            percent = min(100, block * blocksize * 100 / totalsize)
            bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
            print(f"\r   {bar} {percent:.0f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print(f"\n   ✅ 下载完成 ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n   ❌ 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("PP-OCRv6_medium ONNX 模型下载")
    print("=" * 60)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\n📂 模型目录: {models_dir}")
    print()

    success = True
    for model_type, info in MODEL_FILES.items():
        model_dir = models_dir / info["subdir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = model_dir / "inference.onnx"

        if onnx_file.exists():
            size_mb = onnx_file.stat().st_size / 1024 / 1024
            print(f"⏭️   {info['subdir']} 已存在 ({size_mb:.1f} MB)，跳过")
            continue

        if not download_file(info["url"], onnx_file, f"{model_type.upper()}: {info['subdir']}"):
            success = False

    print()
    if success:
        print("✅ 所有模型下载完成！")
        print()
        print("启动服务时，PaddleX 会从 models/ 目录加载模型：")
        print(f"  python server.py")
        print()
        print("如果 PaddleX 仍然使用默认模型，需要设置环境变量：")
        print(f"  export PADDLEX_MODEL_DIR={models_dir}")
    else:
        print("⚠️  部分模型下载失败，请检查网络后重试。")
        print("如果 Hugging Face 无法访问，可以手动下载：")
        print("  https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium")
        print()
        print("下载后将 .onnx 文件放入 models/ 对应子目录。")


if __name__ == "__main__":
    main()
