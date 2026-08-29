"""OCR Server 配置 - PaddleOCR 2.x + PaddlePaddle"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型目录（PaddleOCR 首次运行会自动下载模型到此处）
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# PaddleOCR 会自动下载默认模型，不用手动指定
DET_MODEL_DIR = None
REC_MODEL_DIR = None
CLS_MODEL_DIR = None          # 发票不需要方向分类

# PaddleOCR 参数
USE_ONNX = False               # ONNX 模式（ARM 上如果 onnxruntime 正常就能用）
USE_ANGLE_CLS = False         # 发票不需要方向分类
LANG = "ch"                   # 中文
DROP_SCORE = 0.0              # 不过滤低分结果（保留更多文本供后处理）

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# 发票约束
MAX_INVOICES_PER_IMAGE = 18  # 单张照片最多识别发票数量
