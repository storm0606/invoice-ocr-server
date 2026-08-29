# 发票 OCR 识别服务 (Server Edition)

将发票 OCR 识别从 Android App 迁移到服务器端，利用 PaddleOCR ONNX 模式在轻量服务器上运行。

## 为什么做这个项目

- **手机端 OCR 精度不足** — Android 上的 PaddleOCR Lite 模型识别率不够高
- **需要更好的模型** — 服务器端可以使用标准模型（PP-OCRv4），识别精度更高
- **ARM 服务器可行** — Oracle ARM 服务器 (Ampere Altra) 可以通过 ONNX Runtime 运行 PaddleOCR，无需 GPU
- **与现有 App 解耦** — 手机仅负责拍照上传，服务器完成识别

## 架构

```
手机拍照 → HTTP POST (multipart) → FastAPI 服务器 → PaddleOCR(ONNX) → 发票解析 → JSON 返回
```

## 项目结构

```
invoice-ocr-server/
├── server.py                  # FastAPI 服务入口
├── config.py                  # 配置（模型路径、OCR 参数）
├── requirements.txt           # Python 依赖
├── .gitignore
├── ocr/
│   ├── __init__.py
│   ├── engine.py              # PaddleOCR 引擎封装（ONNX 模式）
│   ├── invoice_codes.py       # 发票代码 → (类型, 金额) 查找表（15 种）
│   └── invoice_parser.py      # 发票解析器（聚类、金额提取、交叉校验）
├── tools/
│   ├── __init__.py
│   └── test_local.py          # 本地单图测试工具
└── README.md
```

## 支持的发票类型

### 一卡通（天津通用定额发票）

| 面额 | 发票代码 |
|---|----|
| 10元 | 112002470103 |
| 50元 | 112002470105 |
| 100元 | 112002470106 |

### 普通发票

| 面额 | 发票代码 |
|---|----|
| 2元 | 112002569131 |
| 3元 | 112002569134 |
| 4元 | 112002569135 |
| 5元 | 112002569136 |
| 6元 | 112002569137 |
| 7元 | 112002569138 |
| 8元 | 112002569139 |
| 9元 | 112002569140 |
| 10元 | 112002569141 |
| 20元 | 112002569143 |
| 50元 | 112002569144 |
| 100元 | 112002569145 |

## 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装依赖

```bash
cd invoice-ocr-server
pip install -r requirements.txt
```

> **注意**: 首次运行会自动下载 PaddleOCR 检测和识别模型（约 100MB）。
> 在 ARM 服务器上，PaddleOCR 使用 ONNX Runtime 运行，无需 PaddlePaddle 框架。
> 如果遇到 `onnxruntime` 兼容性问题，可安装 `onnxruntime-silicon`：
> ```bash
> pip install onnxruntime-silicon
> ```

### 本地启动服务

```bash
python server.py
```

服务启动后访问：
- 健康检查：http://localhost:8000/health
- 发票代码列表：http://localhost:8000/api/codes
- API 文档：http://localhost:8000/docs

### 单图测试

```bash
python tools/test_local.py /path/to/invoice.jpg
python tools/test_local.py /path/to/invoice.jpg --pretty   # JSON 格式化输出
python tools/test_local.py /path/to/invoice.jpg --full     # 显示 OCR 原始结果
```

## API 文档

### POST /api/recognize

上传发票照片进行 OCR 识别。

**请求**: `multipart/form-data`
- `file` — 图片文件（jpg/png/jpeg）

**响应示例**:
```json
{
  "success": true,
  "invoices": [
    {
      "type": "一卡通",
      "amount": 50,
      "invoice_number": "03537528",
      "corrected_by_code": true
    },
    {
      "type": "普通发票",
      "amount": 10,
      "invoice_number": "01425698",
      "corrected_by_code": false
    }
  ],
  "total_count": 2,
  "elapsed_ms": 2450,
  "regions_found": 42
}
```

### GET /health

健康检查。

### GET /api/codes

返回支持的发票代码列表。

## 部署到 Oracle ARM 服务器

### 1. 上传项目

```bash
scp -r invoice-ocr-server/ user@oracle-arm:/home/user/
```

### 2. 安装 Python 和依赖

```bash
ssh user@oracle-arm
cd invoice-ocr-server
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 直接启动（开发）
python server.py

# 使用 systemd 保持后台运行
```

### 4. 防火墙配置

确保服务器防火墙允许 8000 端口：
```bash
sudo ufw allow 8000
# 或 iptables / Oracle Cloud 安全列表添加规则
```

## 识别流程

1. **OCR 识别**: PaddleOCR 检测所有文字区域
2. **空间聚类**: 按 XY 坐标将文字聚合成簇（每张发票对应一个簇）
3. **类型判断**: 通过特征文字判断一卡通/普通发票
4. **金额提取**: 从中文大写金额 / ¥ 符号 / 发票代码中提取
5. **号码提取**: 匹配 8 位发票号码
6. **交叉校验**: 用发票代码（12 位）修正类型和金额
7. **去重**: 同一面额只保留一张（用户需求）

## 与 Android App 的区别

| 特性 | Android App | 本服务 |
|------|---------|--------|
| OCR 引擎 | PaddleOCR Lite | PaddleOCR (标准模型) |
| 推理设备 | 手机 ARM CPU | 服务器 ARM CPU |
| 模型精度 | 低（Lite） | 高（PP-OCRv4） |
| 部署模式 | APK 内嵌 | HTTP 服务 |
| 识别方式 | 本地实时 | 上传 + 远程 |
