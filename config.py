"""OCR Server 配置 - PP-OCRv6_medium"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# PP-OCRv6_medium ONNX 模型路径
# 下载脚本: python tools/download_models.py
DET_MODEL_DIR = os.path.join(MODELS_DIR, "det")          # ch_PP-OCRv6_det
REC_MODEL_DIR = os.path.join(MODELS_DIR, "rec_medium")    # ch_PP-OCRv6_rec_medium
CLS_MODEL_DIR = None                                      # 发票不需要方向分类

# ---- PP-OCRv6_medium 模型下载地址 ----
DET_MODEL_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv6/"
    "ch_PP-OCRv6_det_models.onnx"
)
REC_MODEL_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv6/"
    "ch_PP-OCRv6_rec_medium.onnx"
)

# PaddleOCR 参数
USE_ONNX = True              # ARM 服务器必须用 ONNX 模式
USE_ANGLE_CLS = False        # 发票不需要方向分类
LANG = "ch"                  # 中文
REC_SCORE_THRESH = 0.0       # 不过滤低分结果（保留更多文本供后处理）

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# 发票约束
MAX_INVOICES_PER_IMAGE = 18  # 单张照片最多识别发票数量
