"""
OCR 引擎封装 — PaddleX ONNX Runtime + PP-OCRv4_mobile

使用 PaddleX 3.x + onnxruntime 运行在 ARM 服务器上。

关键优化：
- 默认使用 PP-OCRv4_mobile（~15MB）替代 v6_medium（~134MB），速度提升 5-7 倍
- PaddleX 自动将 PaddlePaddle 模型转换为 ONNX 格式
- 图片缩放至 720px，检测时间减少 ~95%
- 检测/识别阈值设为 0.3，保证召回率
- 批处理大小 12，充分利用 12 核 CPU
- 关闭角度分类
- v4 model_dir 不兼容时自动回退 v6_medium
"""

import logging
import time
import os

import numpy as np

from config import (
    ENGINE,
    DEVICE,
    TEXT_REC_SCORE_THRESH,
    DET_DB_THRESH,
    DET_DB_BOX_THRESH,
    REC_BATCH_NUM,
    ONNX_INTRA_THREADS,
    ONNX_INTER_THREADS,
    DET_LIMIT_SIDE_LEN,
    USE_V4_MODEL,
    V4_DET_MODEL_DIR,
    V4_REC_MODEL_DIR,
    PADDLEX_CACHE,
)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleX ONNX Runtime 引擎封装"""

    def __init__(self):
        self._pipeline = None
        self._initialized = False
        self._use_v4 = False  # tracks which model is actually loaded

    def initialize(self):
        if self._initialized:
            return

        try:
            from paddlex import create_pipeline

            logger.info("Initializing PaddleX OCR pipeline...")
            logger.info(f"  engine={ENGINE}, device={DEVICE}")
            logger.info(f"  use_v4_model={USE_V4_MODEL}")
            logger.info(f"  threads: intra={ONNX_INTRA_THREADS}, inter={ONNX_INTER_THREADS}")
            logger.info(f"  det_thresh={DET_DB_THRESH}, rec_thresh={TEXT_REC_SCORE_THRESH}")
            logger.info(f"  det_limit_side_len={DET_LIMIT_SIDE_LEN}, rec_batch_num={REC_BATCH_NUM}")
            logger.info(f"  use_angle_cls=False")

            # 检查 PaddleX 模型缓存中的可用模型
            self._log_available_models()

            t0 = time.time()

            # 设置 ONNX Runtime 线程环境变量
            os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(ONNX_INTRA_THREADS)
            os.environ["ORT_INTER_OP_NUM_THREADS"] = str(ONNX_INTER_THREADS)

            # 构建 create_pipeline 参数
            pipeline_kwargs = dict(
                pipeline="OCR",
                engine=ENGINE,
                device=DEVICE,
                det_db_thresh=DET_DB_THRESH,
                det_db_box_thresh=DET_DB_BOX_THRESH,
                det_limit_side_len=DET_LIMIT_SIDE_LEN,
                det_limit_type="min",
                rec_batch_num=REC_BATCH_NUM,
                use_angle_cls=False,
            )

            # 尝试使用 PP-OCRv4_mobile 模型
            if USE_V4_MODEL:
                model_dir_ok = os.path.isdir(V4_DET_MODEL_DIR) and os.path.isdir(V4_REC_MODEL_DIR)
                if model_dir_ok:
                    logger.info(f"  ✓ PP-OCRv4_mobile 模型已缓存")
                    logger.info(f"  det_model_dir={V4_DET_MODEL_DIR}")
                    logger.info(f"  rec_model_dir={V4_REC_MODEL_DIR}")
                    pipeline_kwargs["det_model_dir"] = V4_DET_MODEL_DIR
                    pipeline_kwargs["rec_model_dir"] = V4_REC_MODEL_DIR
                else:
                    logger.warning(f"  ✗ PP-OCRv4_mobile 模型未找到，使用默认 v6_medium")
                    if not os.path.isdir(V4_DET_MODEL_DIR):
                        logger.warning(f"    缺少: {V4_DET_MODEL_DIR}")
                    if not os.path.isdir(V4_REC_MODEL_DIR):
                        logger.warning(f"    缺少: {V4_REC_MODEL_DIR}")

            # 创建 pipeline（v4 model_dir 不兼容时自动回退默认模型）
            try:
                self._pipeline = create_pipeline(**pipeline_kwargs)
                self._use_v4 = "det_model_dir" in pipeline_kwargs
            except Exception as v4_err:
                err_str = str(v4_err).lower()
                is_kwarg_err = (
                    "model_dir" in err_str
                    or "unexpected keyword" in err_str
                    or "unexpected argument" in err_str
                    or "got an unexpected" in err_str
                )
                if is_kwarg_err and "det_model_dir" in pipeline_kwargs:
                    logger.warning(f"PP-OCRv4_mobile 模型目录参数不被支持: {v4_err}")
                    logger.warning("回退到 PP-OCRv6_medium 默认模型...")
                    pipeline_kwargs.pop("det_model_dir", None)
                    pipeline_kwargs.pop("rec_model_dir", None)
                    self._pipeline = create_pipeline(**pipeline_kwargs)
                    self._use_v4 = False
                else:
                    raise

            elapsed = time.time() - t0
            model_name = "PP-OCRv4_mobile" if self._use_v4 else "PP-OCRv6_medium"
            logger.info(f"PaddleX pipeline ({model_name}) initialized in {elapsed:.1f}s")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize PaddleX pipeline: {e}")
            raise

    def _log_available_models(self):
        """列出 PaddleX 缓存中的所有可用模型（调试用）"""
        if not os.path.isdir(PADDLEX_CACHE):
            logger.info(f"  PaddleX cache not found: {PADDLEX_CACHE}")
            return
        try:
            models = sorted(os.listdir(PADDLEX_CACHE))
            v4 = [m for m in models if "v4" in m.lower()]
            v6 = [m for m in models if "v6" in m.lower()]
            onnx_models = [m for m in models if "onnx" in m.lower()]
            logger.info(f"  PaddleX cache: {len(models)} models total / "
                        f"{len(v4)} v4 / {len(v6)} v6 / {len(onnx_models)} onnx")
            for m in v4 + v6:
                path = os.path.join(PADDLEX_CACHE, m)
                size = self._dir_size(path)
                logger.info(f"    {m} ({size:.0f} MB)")
        except Exception:
            pass

    @staticmethod
    def _dir_size(path: str) -> float:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total / (1024 * 1024)

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

            # PaddleX OCRResult.json returns {"res": { ... actual data ... }}
            if isinstance(j, dict) and "res" in j:
                inner = j["res"]
            elif isinstance(j, dict):
                inner = j
            else:
                inner = {}

            rec_texts = inner.get("rec_texts", [])
            scores = inner.get("rec_scores", [])
            total_regions = len(rec_texts)

            for i in range(total_regions):
                text = str(rec_texts[i]) if rec_texts[i] else ""
                confidence = 0.0
                if i < len(scores):
                    try:
                        confidence = float(scores[i])
                    except (ValueError, TypeError):
                        pass

                # 过滤低置信度结果
                if confidence < TEXT_REC_SCORE_THRESH:
                    continue

                box = self._get_box(inner, i)
                parsed.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                })

            # 置信度降序排列
            parsed.sort(key=lambda x: x["confidence"], reverse=True)

            # 详细耗时日志
            avg_ms = elapsed * 1000 / total_regions if total_regions else 0
            logger.info(
                f"OCR: {len(parsed)}/{total_regions} regions in {elapsed:.2f}s "
                f"(avg {avg_ms:.0f}ms/region)"
            )
        else:
            logger.info(f"OCR: no results in {elapsed:.2f}s")

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
        """
        # 1. dt_polys: list of lists, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dt_polys = inner.get("dt_polys", [])
        if isinstance(dt_polys, (list, tuple)) and index < len(dt_polys):
            poly = dt_polys[index]
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

        return [[0, 0], [0, 0], [0, 0], [0, 0]]

    def warmup(self):
        logger.info("Warming up OCR engine...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.info("Warmup complete.")

    def close(self):
        self._pipeline = None
        self._initialized = False
