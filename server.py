"""
发票 OCR 识别服务

提供 REST API，接收手机上传的发票照片，
使用 PaddleOCR 进行识别，返回结构化发票信息。

运行:
    python server.py
    # 或
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import HOST, PORT, MAX_IMAGE_SIZE_MB
from ocr.engine import OCREngine
from ocr.invoice_parser import parse_invoices
from ocr.invoice_codes import INVOICE_CODE_MAP

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-server")

# ---------- FastAPI ----------
app = FastAPI(
    title="发票 OCR 识别服务",
    description="接收手机上传的发票照片，使用 PaddleOCR 识别并返回结构化发票信息",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 全局 OCR 引擎 ----------
ocr_engine = OCREngine()


# ---------- 启动事件 ----------
@app.on_event("startup")
async def startup():
    """服务启动时初始化 OCR 引擎并预热"""
    logger.info("=== 发票 OCR 识别服务启动 ===")
    try:
        ocr_engine.initialize()
        logger.info("OCR 引擎初始化完成")
        # 预热（异步执行）
        # ocr_engine.warmup()
    except Exception as e:
        logger.error(f"OCR 引擎初始化失败: {e}")
        raise


# ---------- API ----------

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "ocr_initialized": ocr_engine._initialized,
        "codes_loaded": len(INVOICE_CODE_MAP),
    }


@app.get("/api/codes")
async def list_codes():
    """返回支持的发票代码列表"""
    return {
        "count": len(INVOICE_CODE_MAP),
        "codes": [
            {"code": k, "type": v.type.value, "amount": v.amount}
            for k, v in INVOICE_CODE_MAP.items()
        ],
    }


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    上传发票照片进行 OCR 识别。

    请求: multipart/form-data, file 字段为图片文件
    支持格式: jpg, jpeg, png, heic

    返回:
        success: bool
        invoices: list[dict] 每张发票的信息
        elapsed_ms: int 识别耗时
        error: str (失败时)
    """
    t_start = time.time()

    # 校验文件
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # 读取文件
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_IMAGE_SIZE_MB}MB)")

    logger.info(f"收到图片: {file.filename} ({len(contents) / 1024:.0f} KB)")

    # 解码图片
    try:
        pil_image = Image.open(io.BytesIO(contents))
        # 转 RGB（PIL 可能加载为 RGBA）
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # 转 numpy (H, W, 3), BGR 格式给 PaddleOCR
        img_array = np.array(pil_image)[:, :, ::-1]  # RGB → BGR
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise HTTPException(400, f"Invalid image: {e}")

    # OCR 识别
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

    # 解析发票
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


# ---------- 启动入口 ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
