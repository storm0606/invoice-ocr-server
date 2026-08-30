#!/bin/bash
# ============================================================
# 发票 OCR 服务器性能优化 - 一键部署脚本
# 在服务器上以 root 用户运行
# 用法: bash deploy_optimize.sh
# ============================================================

set -e

echo "=========================================="
echo " 发票 OCR 服务器 - 性能优化部署"
echo "=========================================="

# 1. 进入项目目录
cd /root/invoice-ocr-server || { echo "❌ 项目目录不存在"; exit 1; }
echo "📂 项目目录: $(pwd)"

# 2. 拉取最新代码
echo ""
echo "📥 拉取最新代码..."
git pull origin main
echo "✅ 代码已更新"

# 3. 激活虚拟环境
echo ""
echo "🐍 设置 Python 环境..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "⚠️  未找到虚拟环境，创建中..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ 虚拟环境已创建"
fi

# 4. 安装/更新依赖
echo ""
echo "📦 安装依赖..."
pip install --upgrade pip -q
pip install -r requirements.txt --force-reinstall -q
echo "✅ 依赖安装完成"

# 5. 记录关键包版本
echo ""
echo "📋 关键包版本:"
python3 -c "import paddlex; print(f'  PaddleX: {paddlex.__version__}')" 2>/dev/null || echo "  PaddleX: 需要安装"
python3 -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')" 2>/dev/null || echo "  ONNX Runtime: 需要安装"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: 需要安装"

# 6. 验证配置
echo ""
echo "🔍 验证配置..."
python3 -c "
from config import *
print(f'  MAX_IMAGE_DIM={MAX_IMAGE_DIM}')
print(f'  DET_DB_THRESH={DET_DB_THRESH}')
print(f'  TEXT_REC_SCORE_THRESH={TEXT_REC_SCORE_THRESH}')
print(f'  REC_BATCH_NUM={REC_BATCH_NUM}')
print(f'  ONNX_INTRA_THREADS={ONNX_INTRA_THREADS}')
print(f'  ONNX_INTER_THREADS={ONNX_INTER_THREADS}')
print(f'  DET_LIMIT_SIDE_LEN={DET_LIMIT_SIDE_LEN}')
print(f'  ENGINE={ENGINE}')
print(f'  DEVICE={DEVICE}')
"
echo "✅ 配置验证通过"

# 7. 测试导入（不推理）
echo ""
echo "🧪 测试引擎导入..."
python3 -c "
from ocr.engine import OCREngine
print('  OCREngine 导入成功')
from ocr.invoice_parser import parse_invoices
print('  InvoiceParser 导入成功')
from ocr.invoice_codes import INVOICE_CODE_MAP
print(f'  发票代码表: {len(INVOICE_CODE_MAP)} 种')
"
echo "✅ 引擎导入成功"

# 8. 停止旧服务
echo ""
echo "🛑 停止旧服务..."
if systemctl is-active --quiet invoice-ocr 2>/dev/null; then
    systemctl stop invoice-ocr
    echo "✅ 服务已停止"
else
    # 尝试 kill 进程
    pkill -f "server_ssl.py" 2>/dev/null || true
    pkill -f "server.py" 2>/dev/null || true
    echo "✅ 旧进程已终止"
fi

# 9. 更新 systemd 服务（如果有）
if [ -f /etc/systemd/system/invoice-ocr.service ]; then
    echo ""
    echo "⚙️  更新 systemd 服务..."
    systemctl daemon-reload
    echo "✅ systemd 已更新"
fi

# 10. 启动服务
echo ""
echo "🚀 启动优化后的服务..."
if systemctl list-units --type=service 2>/dev/null | grep -q invoice-ocr; then
    systemctl start invoice-ocr
    systemctl status invoice-ocr --no-pager -l | head -20
else
    echo "使用 nohup 启动..."
    nohup python3 server_ssl.py > server.log 2>&1 &
    echo "PID: $!"
    sleep 3
fi

# 11. 验证服务
echo ""
echo "🏥 验证服务健康..."
for i in 1 2 3 4 5; do
    sleep 2
    RESP=$(curl -sk https://localhost:8443/health 2>/dev/null || curl -s http://localhost:8000/health 2>/dev/null || echo "waiting...")
    echo "  尝试 $i: $RESP"
    if echo "$RESP" | grep -q '"status":"ok"'; then
        echo "✅ 服务启动成功!"
        break
    fi
done

echo ""
echo "=========================================="
echo " 优化部署完成!"
echo "=========================================="
echo ""
echo "📝 实时查看日志:"
echo "  systemctl status invoice-ocr -f"
echo "  或: tail -f server.log"
echo ""
echo "🧪 测试识别:"
echo "  curl -sk -X POST https://localhost:8443/api/recognize -F \"file=@test.jpg\" | python3 -m json.tool"
echo ""
echo "⚡ 优化内容:"
echo "  - 图片缩放: 1200px → 960px"
echo "  - ONNX 线程: intra=12, inter=2"
echo "  - 检测阈值: 0.3 (过滤模糊文字)"
echo "  - 识别阈值: 0.3 (过滤低分结果)"
echo "  - 批处理: 6 个文本/批"
echo "  - 角度分类: 关闭"
echo "=========================================="
