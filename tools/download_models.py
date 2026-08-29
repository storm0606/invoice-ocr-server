#!/usr/bin/env python3
"""
下载 OCR 模型。

默认 PaddleOCR 会自动下载 PP-OCRv4 ONNX 模型（首次运行 server.py 时自动完成），
不需要手动运行此脚本。

如果后续需要使用 PP-OCRv6_medium 模型（来自 Hugging Face）:
  1. 下载: https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium
  2. 解压 ONNX 文件到 models/ 目录
  3. 修改 config.py 中的 DET_MODEL_DIR / REC_MODEL_DIR
"""

import sys


def main():
    print("=" * 60)
    print("模型下载说明")
    print("=" * 60)
    print()
    print("PaddleOCR 会自动下载 PP-OCRv4 ONNX 模型（约 100MB）")
    print("首次启动 server.py 时自动完成，无需手动下载。")
    print()
    print("直接启动服务即可：")
    print("  python server.py")
    print()
    print("如果需要使用 PP-OCRv6_medium 模型：")
    print("  1. 从 Hugging Face 下载 ONNX 文件")
    print("     https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium")
    print("  2. 将 .onnx 文件放到 models/ 目录下")
    print("  3. 修改 config.py 中模型路径配置")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
