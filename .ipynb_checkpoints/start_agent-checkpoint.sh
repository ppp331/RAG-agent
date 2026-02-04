#!/bin/bash
echo "=== 科研流程智能体启动脚本 ==="

# 检查依赖
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python 3.8+"
    exit 1
fi

# 检查必要目录
echo "检查必要目录..."
mkdir -p ./data
mkdir -p ./models

# 检查知识库文件
if [ ! -f "./data/knowledge_db.json" ]; then
    echo "📝 创建默认知识库..."
    echo '[
        {
            "id": 1,
            "type": "protein_workflow",
            "tags": ["蛋白质", "结构预测", "3D可视化", "PDB"],
            "content": "用户输入蛋白质序列（单条或多条）→ 验证序列有效性 → 调用 API 预测结构 → 展示 3D 结构、氨基酸分布和 Ramachandran 图 → 提供 PDB 文件下载"
        }
    ]' > ./data/knowledge_db.json
fi

# 检查配置文件
if [ ! -f "config.py" ]; then
    echo "❌ config.py 文件不存在"
    exit 1
fi

# 检查主文件
if [ ! -f "main.py" ]; then
    echo "❌ main.py 文件不存在"
    exit 1
fi

echo "🚀 启动科研流程智能体..."
echo "使用模型: DeepSeek + 百度文心千帆Embedding"
echo "-" * 50

python3 main.py