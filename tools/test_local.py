#!/usr/bin/env python3
"""
本地测试工具 - 对单张发票照片执行 OCR 识别并打印结果。

用法:
    python tools/test_local.py /path/to/invoice.jpg

依赖:
    pip install -r requirements.txt

注意: 首次运行会下载 PaddleOCR 模型（约 100MB），需要联网。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from ocr.engine import OCREngine
from ocr.invoice_parser import parse_invoices


def main():
    parser = argparse.ArgumentParser(
        description="对单张发票照片执行 OCR 识别并打印结构化结果",
    )
    parser.add_argument("image_path", type=str, help="发票照片路径")
    parser.add_argument(
        "--pretty", "-p", action="store_true",
        help="结构化输出（JSON 缩进）",
    )
    parser.add_argument(
        "--full", "-f", action="store_true",
        help="打印 OCR 原始结果（所有文本区域）",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ 文件不存在: {image_path}")
        sys.exit(1)

    # ---- 加载图片 ----
    print(f"📷 加载图片: {image_path}")
    pil_image = Image.open(image_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_array = np.array(pil_image)[:, :, ::-1]  # RGB → BGR
    print(f"   尺寸: {pil_image.width} × {pil_image.height}")
    print()

    # ---- OCR 识别 ----
    print("🔍 初始化 PaddleOCR (首次运行会下载模型)...")
    t0 = time.time()
    engine = OCREngine()
    engine.initialize()
    print(f"   初始化耗时: {time.time() - t0:.1f}s")

    print("🔍 运行 OCR...")
    t1 = time.time()
    raw = engine.recognize(img_array)
    ocr_ms = int((time.time() - t1) * 1000)
    print(f"   OCR 耗时: {ocr_ms}ms")
    print()

    if not raw:
        print("⚠️  未识别到任何文本")
        sys.exit(0)

    if args.full:
        print("📄 OCR 原始结果:")
        for i, r in enumerate(raw):
            print(f"  [{i:02d}] \"{r['text']}\" (conf={r['confidence']:.3f}) "
                  f"box=({r['box'][0][0]},{r['box'][0][1]})")
        print()

    # ---- 解析发票 ----
    print("📊 发票解析...")
    t2 = time.time()
    invoices = parse_invoices(raw)
    parse_ms = int((time.time() - t2) * 1000)
    total_ms = int((time.time() - t1) * 1000)

    # ---- 输出结果 ----
    if args.pretty:
        print(json.dumps({
            "success": True,
            "total_count": len(invoices),
            "elapsed_ms": total_ms,
            "ocr_regions": len(raw),
            "invoices": invoices,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"✅ 识别到 {len(invoices)} 张发票 (OCR: {ocr_ms}ms, 解析: {parse_ms}ms, 总计: {total_ms}ms)")
        print()
        if not invoices:
            print("  无发票识别结果")
        else:
            for inv in invoices:
                mark = "✓" if not inv.get("corrected_by_code") else "✓(代码)"
                print(f"  {mark} {inv['type']} {inv['amount']}元 "
                      f"  号码={inv['invoice_number']}")


if __name__ == "__main__":
    main()
