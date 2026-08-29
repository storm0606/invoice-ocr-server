"""OCR Server 配置 - PaddleX ONNX Runtime"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# PaddleX 3.x ONNX Runtime 参数
LANG = "ch"                          # 中文
OCR_VERSION = "PP-OCRv6_medium"      # OCR 版本（ONNX 模型）
TEXT_REC_SCORE_THRESH = 0.0          # 不过滤低分结果

# 检测参数
TEXT_DET_LIMIT_SIDE_LEN = 2000       # 最大图片边长
TEXT_DET_LIMIT_TYPE = "min"          # "min"=按最短边缩放

# PaddleX 引擎设置
ENGINE = "onnxruntime"               # 推理引擎：onnxruntime（ARM 稳定）
DEVICE = "cpu"                        # 设备

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# 发票约束
MAX_INVOICES_PER_IMAGE = 18
