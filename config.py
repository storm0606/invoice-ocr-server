"""OCR Server 配置 — PaddleX ONNX Runtime + PP-OCRv4_mobile"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== 模型选择 ====================

# 默认使用 PP-OCRv4_mobile（~15MB）而非 v6_medium（~134MB）
# v4_mobile 在 ARM CPU 上推理速度约快 5-7 倍
USE_V4_MODEL = True

# PaddleX 缓存中的模型目录
PADDLEX_CACHE = os.path.expanduser("~/.paddlex/official_models")

# PP-OCRv4_mobile 模型路径（PaddlePaddle 格式，PaddleX 会自动转 ONNX）
V4_DET_MODEL_DIR = os.path.join(PADDLEX_CACHE, "PP-OCRv4_mobile_det")
V4_REC_MODEL_DIR = os.path.join(PADDLEX_CACHE, "PP-OCRv4_mobile_rec")

# 检测模型输入最大边长（越小越快）
#  960 → 720 约减少 44% 像素，检测速度提升 ~40%
DET_LIMIT_SIDE_LEN = 720

# 检测置信度阈值（0.3 能捕捉更多模糊文本区域）
DET_DB_THRESH = 0.3
# 检测框置信度阈值（0.5 过滤过于歪斜的框）
DET_DB_BOX_THRESH = 0.5

# 识别置信度阈值（0.3 保留低分候选，配合后续代码校验）
TEXT_REC_SCORE_THRESH = 0.3

# 识别批处理大小（越大越快，12 对 ARM 12 核 CPU 较优）
REC_BATCH_NUM = 12

# ==================== 引擎设置 ====================

ENGINE = "onnxruntime"   # onnxruntime（ARM 稳定）
DEVICE = "cpu"

# ONNX Runtime 线程数（根据 CPU 核数调整）
ONNX_INTRA_THREADS = 12   # 单算子内并行（= 物理核数）
ONNX_INTER_THREADS = 2    # 算子间并行

os.environ.setdefault("OMP_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", str(ONNX_INTER_THREADS))

# ==================== 图片预处理 ====================

# 手机照片缩放最大边长（降低此值即可大幅提速）
#  原始: 3000-4000px → 720px，面积减少 ~95%，检测速度提升 ~10x
MAX_IMAGE_DIM = 720

# ==================== 服务器配置 ====================

HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10
MAX_INVOICES_PER_IMAGE = 30
