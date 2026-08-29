"""
OCR 引擎封装 - PaddleX ONNX Runtime

使用 PaddleX 3.x + onnxruntime 运行在 ARM 服务器上。
避免了 PaddlePaddle 原生推理在 ARM 上的 segfault 问题。
"""

import logging
import time
import os

import numpy as np

from config import (
    ENGINE,
    DEVICE,
    TEXT_REC_SCORE_THRESH,
)

# ARM 上限制线程数
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleX ONNX Runtime 引擎封装"""

    def __init__(self):
        self._pipeline = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        try:
            from paddlex import create_pipeline

            logger.info("Initializing PaddleX OCR pipeline...")
            logger.info(f"  engine={ENGINE}, device={DEVICE}")

            t0 = time.time()
            self._pipeline = create_pipeline(
                pipeline="OCR",
                engine=ENGINE,
                device=DEVICE,
            )
            logger.info(f"PaddleX pipeline initialized in {time.time() - t0:.1f}s")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize PaddleX pipeline: {e}")
            raise

    def recognize(self, image: np.ndarray) -> list[dict]:
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        try:
            results = list(self._pipeline(image))
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return []

        elapsed = time.time() - t0
        parsed = []

        if results:
            # OCRResult 通过 .json 属性获取 dict
            r = results[0]
            j = r.json if hasattr(r, 'json') else {}
            if callable(j):
                j = j()

            rec_texts = j.get("rec_texts", []) if isinstance(j, dict) else []
            rec_scores = j.get("rec_scores", []) if isinstance(j, dict) else []
            rec_boxes = j.get("rec_boxes", []) if isinstance(j, dict) else []

            for i in range(len(rec_texts)):
                text = str(rec_texts[i]) if rec_texts[i] else ""
                confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                box = self._normalize_box(rec_boxes[i]) if i < len(rec_boxes) else []
                parsed.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                })

        logger.info(f"OCR: {len(parsed)} text regions in {elapsed:.2f}s")
        return parsed

    def _normalize_box(self, box) -> list:
        if not box:
            return []
        if isinstance(box, np.ndarray):
            if box.shape == (4, 2):
                return [[int(p[0]), int(p[1])] for p in box]
            if box.shape == (4,):
                x1, y1, x2, y2 = [int(v) for v in box]
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            return []
        if isinstance(box, (list, tuple)):
            if len(box) == 4 and all(isinstance(p, (int, float)) for p in box):
                x1, y1, x2, y2 = [int(v) for v in box]
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if len(box) == 4 and all(isinstance(p, (list, tuple)) for p in box):
                return [[int(p[0]), int(p[1])] for p in box]
        return []

    def warmup(self):
        logger.info("Warming up OCR engine...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.info("Warmup complete.")

    def close(self):
        self._pipeline = None
        self._initialized = False
