"""OCR Server 配置 - PaddleOCR 3.x"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型目录（PaddleOCR 首次运行会自动下载模型）
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# PaddleOCR 会自动下载默认模型，不用手动指定
# 如需使用本地 ONNX 模型，可设置:
#   text_detection_model_dir = os.path.join(MODELS_DIR, "det")
#   text_recognition_model_dir = os.path.join(MODELS_DIR, "rec_medium")
TEXT_DETECTION_MODEL_DIR = None
TEXT_RECOGNITION_MODEL_DIR = None

# PaddleOCR 3.x 参数
LANG = "ch"                          # 中文
OCR_VERSION = "PP-OCRv4"             # OCR 版本
TEXT_REC_SCORE_THRESH = 0.0          # 不过滤低分结果
USE_DOC_ORIENTATION_CLASSIFY = False  # 发票不需要方向分类
USE_DOC_UNWARPING = False             # 发票不需要文档拉直
USE_TEXTLINE_ORIENTATION = False      # 发票不需要文字行方向

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# 发票约束
MAX_INVOICES_PER_IMAGE = 18  # 单张照片最多识别发票数量
