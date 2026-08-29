"""
OCR 引擎封装。

使用 PaddleOCR 2.x + PaddlePaddle 运行在 ARM 服务器上。
"""

import logging
import time
from pathlib import Path

import numpy as np

from config import (
    DET_MODEL_DIR,
    REC_MODEL_DIR,
    CLS_MODEL_DIR,
    USE_ONNX,
    USE_ANGLE_CLS,
    LANG,
    DROP_SCORE,
)

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR 引擎的轻量封装"""

    def __init__(self):
        self._ocr = None
        self._initialized = False

    def initialize(self):
        """初始化 PaddleOCR（首次调用会下载模型）"""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            kwargs = {
                "use_angle_cls": USE_ANGLE_CLS,
                "lang": LANG,
                "use_onnx": USE_ONNX,
                "show_log": False,
                "drop_score": DROP_SCORE,
            }

            if DET_MODEL_DIR:
                kwargs["det_model_dir"] = DET_MODEL_DIR
            if REC_MODEL_DIR:
                kwargs["rec_model_dir"] = REC_MODEL_DIR
            if CLS_MODEL_DIR:
                kwargs["cls_model_dir"] = CLS_MODEL_DIR

            logger.info("Initializing PaddleOCR...")
            logger.info(f"  use_onnx={USE_ONNX}, lang={LANG}")
            logger.info(f"  det_model_dir={DET_MODEL_DIR}")
            logger.info(f"  rec_model_dir={REC_MODEL_DIR}")

            t0 = time.time()
            self._ocr = PaddleOCR(**kwargs)
            logger.info(f"PaddleOCR initialized in {time.time() - t0:.1f}s")
            self._initialized = True

        except ImportError as e:
            logger.error(f"Failed to import PaddleOCR: {e}")
            logger.error("Run: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def recognize(self, image: np.ndarray) -> list[dict]:
        """
        对图像进行 OCR 识别。

        Args:
            image: numpy array (H, W, 3), BGR 格式

        Returns:
            list[dict]: 每个元素包含:
                - text: str, 识别文本
                - confidence: float, 置信度
                - box: list[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], 文本框坐标
        """
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        try:
            results = self._ocr.ocr(image, cls=False)
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return []

        elapsed = time.time() - t0
        parsed = []

        if results and results[0]:
            for line in results[0]:
                box, (text, confidence) = line
                parsed.append({
                    "text": text,
                    "confidence": confidence,
                    "box": [[int(p[0]), int(p[1])] for p in box],
                })

        logger.info(f"OCR: {len(parsed)} text regions in {elapsed:.2f}s")
        return parsed

    def warmup(self):
        """预热：用空图跑一次推理，加载模型到内存"""
        logger.info("Warming up OCR engine...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.info("Warmup complete.")

    def close(self):
        """释放资源"""
        self._ocr = None
        self._initialized = False
