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
            # PaddleX OCRResult.json 返回 {"res": { ... 实际数据 ... }}
            # 数据嵌套在 "res" 里面，不能直接在顶层取 rec_texts
            r = results[0]
            j = r.json if hasattr(r, "json") else {}
            if callable(j):
                j = j()

            # 解包 res 层
            if isinstance(j, dict) and "res" in j:
                inner = j["res"]
            elif isinstance(j, dict):
                inner = j
            else:
                inner = {}

            rec_texts = inner.get("rec_texts", [])
            rec_scores = inner.get("rec_scores", [])
            # rec_boxes 是 (N, 4) 数组 [x1, y1, x2, y2]
            rec_boxes = inner.get("rec_boxes", [])

            for i in range(len(rec_texts)):
                text = str(rec_texts[i]) if rec_texts[i] else ""
                confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                box = self._normalize_rec_box(rec_boxes, i)
                parsed.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                })

            # 按置信度降序排列
            parsed.sort(key=lambda x: x["confidence"], reverse=True)

            # 打印所有文本区域（调试用）
            logger.info(f"OCR 文本区域详情 ({len(parsed)}):")

        logger.info(f"OCR: {len(parsed)} text regions in {elapsed:.2f}s")
        return parsed

    @staticmethod
    def _normalize_rec_box(rec_boxes, index: int) -> list:
        """
        rec_boxes 是 (N, 4) 数组 [x1, y1, x2, y2] 或 (N, 4, 2) 多边形。
        统一输出为 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]。
        """
        if isinstance(rec_boxes, np.ndarray):
            if rec_boxes.ndim == 2 and rec_boxes.shape[1] == 4:
                x1, y1, x2, y2 = rec_boxes[index]
                return [[int(x1), int(y1)], [int(x2), int(y1)],
                        [int(x2), int(y2)], [int(x1), int(y2)]]
        if isinstance(rec_boxes, (list, tuple)) and index < len(rec_boxes):
            box = rec_boxes[index]
            if isinstance(box, np.ndarray) and box.shape == (4, 2):
                return [[int(box[p][0]), int(box[p][1])] for p in range(4)]
        return []

    def warmup(self):
        logger.info("Warming up OCR engine...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.info("Warmup complete.")

    def close(self):
        self._pipeline = None
        self._initialized = False
