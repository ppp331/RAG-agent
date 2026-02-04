#!/bin/bash
echo "=== 科研流程智能体启动脚本 ==="
echo "使用DeepSeek API版本"

# 检查Python环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python 3.8+"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
python -c "import sentence_transformers, sklearn, requests, numpy, autogen" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip install -r requirements.txt 2>/dev/null || {
        echo "❌ 依赖安装失败，请手动安装"
        echo "   运行: pip install sentence-transformers scikit-learn numpy requests pyautogen"
        exit 1
    }
fi

# 检查嵌入模型
echo "检查嵌入模型..."
if [ ! -d "./models/all-MiniLM-L6-v2" ]; then
    echo "📥 下载嵌入模型..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "❌ 模型下载失败，尝试使用国内镜像..."
        export HF_ENDPOINT=https://hf-mirror.com
        python download_model.py
    fi
fi

# 启动智能体
echo "🚀 启动科研流程智能体..."
echo "使用模型: DeepSeek Chat"
echo "API配置检查..."
python check_config.py

if [ $? -eq 0 ]; then
    echo ""
    echo "正在启动主程序..."
    python main.py
else
    echo "❌ 环境检查失败，请修复后重试"
    exit 1
fi