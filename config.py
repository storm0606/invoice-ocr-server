"""OCR Server 配置"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---- 默认行为: 让 PaddleOCR 自动下载 ONNX 模型 ----
# PaddleOCR 在 use_onnx=True 模式下会自动下载 PP-OCRv4 的 ONNX 模型
DET_MODEL_DIR = None
REC_MODEL_DIR = None
CLS_MODEL_DIR = None          # 发票不需要方向分类

# ---- 如果要用 PP-OCRv6_medium 模型，取消下面注释并设回 None ----
# 下载方式: 从 Hugging Face 下载 PP-OCRv6_rec_medium.onnx
# cnocr: https://huggingface.co/breezedeus/cnocr-ppocr-multi_PP-OCRv6_medium
# DET_MODEL_DIR = os.path.join(MODELS_DIR, "det")
# REC_MODEL_DIR = os.path.join(MODELS_DIR, "rec_medium")

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
