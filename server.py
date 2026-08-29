"""
发票 OCR 识别服务

提供 REST API，接收手机上传的发票照片，
使用 PaddleX + ONNX Runtime 进行识别，返回结构化发票信息。

运行:
    python server.py
"""

import io
import logging
import time

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import HOST, PORT, MAX_IMAGE_SIZE_MB
from ocr.engine import OCREngine
from ocr.invoice_parser import parse_invoices
from ocr.invoice_codes import INVOICE_CODE_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-server")

app = FastAPI(
    title="发票 OCR 识别服务",
    description="接收手机上传的发票照片，使用 PaddleOCR 识别并返回结构化发票信息",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_engine = OCREngine()

# ARM 服务器性能有限，限制图片尺寸以加快推理
MAX_IMAGE_DIM = 1200


def resize_if_needed(pil_image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    w, h = pil_image.size
    if max(w, h) <= max_dim:
        return pil_image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"图片缩放: {w}x{h} -> {new_w}x{new_h}")
    return pil_image.resize((new_w, new_h), Image.LANCZOS)


@app.on_event("startup")
async def startup():
    logger.info("=== 发票 OCR 识别服务启动 ===")
    try:
        ocr_engine.initialize()
        logger.info("OCR 引擎初始化完成")
    except Exception as e:
        logger.error(f"OCR 引擎初始化失败: {e}")
        raise


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ocr_initialized": ocr_engine._initialized,
        "codes_loaded": len(INVOICE_CODE_MAP),
    }


@app.get("/api/codes")
async def list_codes():
    return {
        "count": len(INVOICE_CODE_MAP),
        "codes": [
            {"code": k, "type": v.type.value, "amount": v.amount}
            for k, v in INVOICE_CODE_MAP.items()
        ],
    }


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    t_start = time.time()

    if not file.filename:
        raise HTTPException(400, "No file provided")

    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_IMAGE_SIZE_MB}MB)")

    logger.info(f"收到图片: {file.filename} ({len(contents) / 1024:.0f} KB)")

    try:
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image = resize_if_needed(pil_image)
        # RGB -> BGR for PaddleX
        img_array = np.array(pil_image)[:, :, ::-1]
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        ocr_results = ocr_engine.recognize(img_array)
    except Exception as e:
        logger.error(f"OCR 识别失败: {e}")
        return {
            "success": False,
            "error": f"OCR failed: {e}",
            "invoices": [],
            "elapsed_ms": int((time.time() - t_start) * 1000),
        }

    if not ocr_results:
        logger.warning("OCR 未识别到任何文本")
        return {
            "success": True,
            "invoices": [],
            "elapsed_ms": int((time.time() - t_start) * 1000),
            "message": "No text recognized in image",
        }

    try:
        invoices = parse_invoices(ocr_results)
    except Exception as e:
        logger.error(f"发票解析失败: {e}")
        invoices = []

    elapsed_ms = int((time.time() - t_start) * 1000)
    logger.info(f"总耗时: {elapsed_ms}ms, 识别 {len(invoices)} 张发票")

    return {
        "success": True,
        "invoices": invoices,
        "total_count": len(invoices),
        "elapsed_ms": elapsed_ms,
        "regions_found": len(ocr_results),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
