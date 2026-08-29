"""OCR Server 配置"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型目录（paddleocr 会自动下载，也可以提前下载放到这里）
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 如果使用 paddleocr 内置模型，设置为 None
# 如果使用自己的 ONNX 模型，设置对应目录
DET_MODEL_DIR = None      # 例如: os.path.join(MODELS_DIR, "det")
REC_MODEL_DIR = None      # 例如: os.path.join(MODELS_DIR, "rec_medium")
CLS_MODEL_DIR = None      # 方向分类模型（发票不需要）

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
