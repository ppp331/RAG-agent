import requests
import subprocess
import sys
from config import Config

def check_deepseek():
    """检查DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        # 简单的模型列表查询
        response = requests.get("https://api.deepseek.com/v1/models", 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ DeepSeek API 连接正常")
            return True
        else:
            print(f"❌ DeepSeek API 连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ DeepSeek API 连接错误: {e}")
        print("   请检查：")
        print("   1. API密钥是否正确")
        print("   2. 网络连接是否正常")
        return False

def check_dependencies():
    """检查Python依赖"""
    dependencies = ['sentence-transformers', 'sklearn', 'requests', 'numpy', 'autogen']
    print("检查Python依赖包...")
    for dep in dependencies:
        try:
            __import__(dep if dep != 'sklearn' else 'sklearn')
            print(f"  ✅ {dep} 已安装")
        except ImportError:
            print(f"  ❌ {dep} 未安装")
            if dep == 'autogen':
                print("     安装: pip install pyautogen")
            elif dep == 'sentence-transformers':
                print("     安装: pip install sentence-transformers")

def check_embedding_model():
    """检查嵌入模型"""
    try:
        from sentence_transformers import SentenceTransformer
        local_model_path = "./models/all-MiniLM-L6-v2"
        if os.path.exists(local_model_path):
            print("✅ 嵌入模型已下载")
            return True
        else:
            print("⚠️  嵌入模型未下载，首次运行时会自动下载")
            return False
    except ImportError:
        print("❌ sentence-transformers 未安装")
        return False

if __name__ == "__main__":
    print("=== 科研流程智能体环境检查 ===\n")
    
    print("1. Python依赖检查:")
    check_dependencies()
    
    print("\n2. API服务检查:")
    deepseek_ok = check_deepseek()
    
    print("\n3. 嵌入模型检查:")
    check_embedding_model()
    
    print("\n" + "="*50)
    
    if deepseek_ok:
        print("✅ 环境检查通过，可以启动智能体")
        print("   启动命令: python main.py")
    else:
        print("❌ DeepSeek API 连接失败，请先修复")