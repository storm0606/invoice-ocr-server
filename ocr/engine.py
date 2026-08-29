"""
OCR 引擎封装 - PaddleOCR 3.x + PaddleX

使用 PaddleOCR 3.x 的 ONNX 支持运行在 ARM 服务器上，
通过 PaddleX 自动选择推理引擎。
"""

import logging
import time

import numpy as np

from config import (
    TEXT_DETECTION_MODEL_DIR,
    TEXT_RECOGNITION_MODEL_DIR,
    LANG,
    OCR_VERSION,
    TEXT_REC_SCORE_THRESH,
    USE_DOC_ORIENTATION_CLASSIFY,
    USE_DOC_UNWARPING,
    USE_TEXTLINE_ORIENTATION,
)

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR 3.x 引擎封装"""

    def __init__(self):
        self._ocr = None
        self._initialized = False

    def initialize(self):
        """初始化 PaddleOCR（首次调用会下载模型）"""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            logger.info("Initializing PaddleOCR 3.x...")
            logger.info(f"  ocr_version={OCR_VERSION}, lang={LANG}")

            kwargs = {
                "lang": LANG,
                "ocr_version": OCR_VERSION,
                "use_doc_orientation_classify": USE_DOC_ORIENTATION_CLASSIFY,
                "use_doc_unwarping": USE_DOC_UNWARPING,
                "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
                "text_rec_score_thresh": TEXT_REC_SCORE_THRESH,
            }

            if TEXT_DETECTION_MODEL_DIR:
                kwargs["text_detection_model_dir"] = TEXT_DETECTION_MODEL_DIR
            if TEXT_RECOGNITION_MODEL_DIR:
                kwargs["text_recognition_model_dir"] = TEXT_RECOGNITION_MODEL_DIR

            t0 = time.time()
            self._ocr = PaddleOCR(**kwargs)
            logger.info(f"PaddleOCR initialized in {time.time() - t0:.1f}s")
            self._initialized = True

        except ImportError as e:
            logger.error(f"Failed to import PaddleOCR: {e}")
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
            # PaddleOCR 3.x 支持 __call__ 或 predict 方法
            # 尝试多种调用方式兼容
            if hasattr(self._ocr, "ocr") and callable(self._ocr.ocr):
                results = self._ocr.ocr(image)
            elif hasattr(self._ocr, "predict") and callable(self._ocr.predict):
                results = self._ocr.predict(image)
            elif callable(self._ocr):
                results = self._ocr(image)
            else:
                raise RuntimeError("No callable method found on PaddleOCR")
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return []

        elapsed = time.time() - t0
        parsed = []

        # PaddleOCR 3.x 返回结构可能是 PaddleX Result 对象
        # 尝试兼容多种返回格式
        try:
            raw_results = self._parse_paddlex_result(results)
        except Exception:
            raw_results = results if results else []

        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    confidence = item.get("confidence", 0.0)
                    box = item.get("box", [])
                    parsed.append({
                        "text": str(text),
                        "confidence": float(confidence),
                        "box": box,
                    })
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    box, (text, confidence) = item
                    parsed.append({
                        "text": str(text),
                        "confidence": float(confidence),
                        "box": [[int(p[0]), int(p[1])] for p in box],
                    })

        logger.info(f"OCR: {len(parsed)} text regions in {elapsed:.2f}s")
        return parsed

    def _parse_paddlex_result(self, result) -> list:
        """尝试解析 PaddleX 结果对象为统一格式"""
        if result is None:
            return []

        # PaddleX Result 对象可能包含 .ocr() 或 .json() 方法
        if hasattr(result, "json"):
            return result.json().get("boxes", result.json())

        if hasattr(result, "ocr"):
            return result.ocr()

        if isinstance(result, dict):
            return result.get("boxes", result.get("results", [result]))

        if isinstance(result, list):
            return result

        return []

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
