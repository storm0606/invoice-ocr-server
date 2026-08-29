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
            logger.error(f"OCR pipeline error: {e}")
            return []

        elapsed = time.time() - t0
        parsed = []

        if results:
            r = results[0]
            j = r.json if hasattr(r, "json") else {}
            if callable(j):
                j = j()

            # PaddleX OCRResult.json 返回 {"res": { ... 实际数据 ... }}
            if isinstance(j, dict) and "res" in j:
                inner = j["res"]
            elif isinstance(j, dict):
                inner = j
            else:
                inner = {}

            rec_texts = inner.get("rec_texts", [])

            for i in range(len(rec_texts)):
                text = str(rec_texts[i]) if rec_texts[i] else ""
                confidence = 0.0
                scores = inner.get("rec_scores", [])
                if i < len(scores):
                    try:
                        confidence = float(scores[i])
                    except (ValueError, TypeError):
                        pass
                box = self._get_box(inner, i)
                parsed.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                })

            # 置信度降序排列
            parsed.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(f"OCR: {len(parsed)} text regions in {elapsed:.2f}s")
        return parsed

    @staticmethod
    def _get_box(inner: dict, index: int) -> list:
        """
        从 inner dict 中提取第 index 个文本区域的边界框。

        优先级：
        1. dt_polys (4 点多边形列表)
        2. rec_boxes (N, 4) [x1, y1, x2, y2] 列表
        3. rec_polys (4 点多边形列表)
        4. 默认返回单位矩形

        PaddleX v3 返回 Python list（不是 numpy ndarray）：
          dt_polys[i] = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
          rec_boxes[i] = [x1, y1, x2, y2]
        """
        # 1. dt_polys: list of lists, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dt_polys = inner.get("dt_polys", [])
        if isinstance(dt_polys, (list, tuple)) and index < len(dt_polys):
            poly = dt_polys[index]
            # numpy ndarray: has .shape; Python list: check length
            if hasattr(poly, 'shape') and poly.shape == (4, 2):
                try:
                    return [[int(poly[p][0]), int(poly[p][1])] for p in range(4)]
                except (ValueError, TypeError, IndexError):
                    pass
            elif isinstance(poly, (list, tuple)) and len(poly) == 4:
                try:
                    return [[int(poly[p][0]), int(poly[p][1])] for p in range(4)]
                except (ValueError, TypeError, IndexError):
                    pass

        # 2. rec_boxes: list of [x1, y1, x2, y2] (ndarray or list)
        rboxes = inner.get("rec_boxes", [])
        if isinstance(rboxes, np.ndarray) and rboxes.ndim == 2 and index < rboxes.shape[0]:
            try:
                row = rboxes[index]
                if row.shape[0] >= 4:
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            except (ValueError, TypeError, IndexError):
                pass
        elif isinstance(rboxes, (list, tuple)) and index < len(rboxes):
            try:
                row = rboxes[index]
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            except (ValueError, TypeError, IndexError):
                pass

        # 3. rec_polys: list of lists (same format as dt_polys)
        rpolys = inner.get("rec_polys", [])
        if isinstance(rpolys, (list, tuple)) and index < len(rpolys):
            poly = rpolys[index]
            if hasattr(poly, 'shape') and poly.shape == (4, 2):
                try:
                    return [[int(poly[p][0]), int(poly[p][1])] for p in range(4)]
                except (ValueError, TypeError, IndexError):
                    pass
            elif isinstance(poly, (list, tuple)) and len(poly) == 4:
                try:
                    return [[int(poly[p][0]), int(poly[p][1])] for p in range(4)]
                except (ValueError, TypeError, IndexError):
                    pass

        # 4. 默认返回单位矩形
        return [[0, 0], [0, 0], [0, 0], [0, 0]]

    def warmup(self):
        logger.info("Warming up OCR engine...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.info("Warmup complete.")

    def close(self):
        self._pipeline = None
        self._initialized = False
