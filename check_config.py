import requests
import subprocess
import sys
import os
from config import Config

def check_deepseek():
    """检查DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.deepseek.com/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ DeepSeek API 连接正常")
            return True
        else:
            print(f"❌ DeepSeek API 连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ DeepSeek API 连接错误: {e}")
        return False

def check_baidu_network():
    """检查百度API网络连接"""
    try:
        # 测试百度API端点
        test_url = "https://aip.baidubce.com"
        response = requests.get(test_url, timeout=5)
        if response.status_code < 500:
            print("✅ 百度API网络连接正常")
            return True
        else:
            print(f"❌ 百度API网络异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 百度API网络连接错误: {e}")
        return False

def check_dependencies():
    """检查Python依赖"""
    dependencies = ['sklearn', 'requests', 'numpy', 'autogen']
    print("检查Python依赖包...")
    for dep in dependencies:
        try:
            __import__(dep if dep != 'sklearn' else 'sklearn')
            print(f"  ✅ {dep} 已安装")
        except ImportError:
            print(f"  ❌ {dep} 未安装")
            if dep == 'autogen':
                print("     安装: pip install pyautogen")
            elif dep == 'sklearn':
                print("     安装: pip install scikit-learn")

if __name__ == "__main__":
    print("=== 科研流程智能体环境检查 ===\n")
    
    print("1. Python依赖检查:")
    check_dependencies()
    
    print("\n2. API服务检查:")
    print(f"   主模型: DeepSeek")
    deepseek_ok = check_deepseek()
    
    print(f"\n3. Embedding服务检查:")
    print(f"   Provider: {Config.EMBEDDING_API_PROVIDER}")
    baidu_network_ok = check_baidu_network()
    
    print("\n4. 配置文件检查:")
    print(f"   DeepSeek模型: {Config.DEEPSEEK_MODEL}")
    print(f"   Embedding模型: {Config.EMBEDDING_MODEL}")
    print(f"   API Key格式: {'有效' if Config.BAIDU_API_KEY.startswith('bce-v3/') else '传统格式'}")
    
    print("\n" + "="*50)
    
    if deepseek_ok and baidu_network_ok:
        print("✅ 环境检查通过，可以启动智能体")
        print("   启动命令: python main.py")
    else:
        print("⚠️  部分检查失败，但程序可能仍可运行")
        print("   启动命令: python main.py")