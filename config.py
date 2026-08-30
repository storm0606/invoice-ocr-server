"""OCR Server 配置 - PaddleX ONNX Runtime"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== OCR 引擎配置 ====================

# PaddleX 3.x ONNX Runtime 参数
LANG = "ch"                          # 中文
OCR_VERSION = "PP-OCRv6_medium"      # OCR 版本（ONNX 模型）

# 检测参数
DET_LIMIT_SIDE_LEN = 960             # 检测模型输入最大边长（越小越快，960 是 PaddleOCR 默认值）
DET_DB_THRESH = 0.3                  # 检测置信度阈值（0.3 过滤掉模糊文本区域）
DET_DB_BOX_THRESH = 0.5              # 检测框置信度阈值

# 识别参数
TEXT_REC_SCORE_THRESH = 0.3          # 识别置信度阈值（过低的结果丢弃，减少误识别）
REC_BATCH_NUM = 6                    # 识别批处理大小（越大越快，但占用更多内存）

# PaddleX 引擎设置
ENGINE = "onnxruntime"               # 推理引擎：onnxruntime（ARM 稳定）
DEVICE = "cpu"                        # 设备

# ONNX Runtime 线程配置（ARM Ampere Altra CPU 优化）
# 12 核 CPU: intra=12, inter=2；4 核 CPU: intra=4, inter=1
ONNX_INTRA_THREADS = 12              # 单算子内并行线程数（= CPU 核数）
ONNX_INTER_THREADS = 2               # 算子间并行线程数

# 环境变量线程限制
os.environ.setdefault("OMP_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(ONNX_INTRA_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ONNX_INTRA_THREADS))

# ==================== 图片预处理 ====================

# 服务器端对上传图片进行缩放的最大边长
# 手机照片通常为 3000-4000px，缩放后大幅减少检测时间
# ARM 服务器建议 960，GPU 服务器可设为 1600+
MAX_IMAGE_DIM = 960

# ==================== 服务器配置 ====================

HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# 发票约束
MAX_INVOICES_PER_IMAGE = 18
