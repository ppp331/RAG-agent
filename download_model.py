from sentence_transformers import SentenceTransformer
import os

def download_embedding_model():
    """单独下载Embedding模型"""
    print("开始下载SentenceTransformer模型...")
    print("模型: all-MiniLM-L6-v2")
    print("这可能需要几分钟，请耐心等待...")
    
    try:
        # 设置镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 下载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 测试模型是否正常工作
        embeddings = model.encode(["测试文本"])
        print(f"✅ 模型下载成功！向量维度: {embeddings.shape}")
        
        # 保存模型到本地路径，避免每次下载
        model_path = "./models/all-MiniLM-L6-v2"
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)
        print(f"✅ 模型已保存到: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        return False

if __name__ == "__main__":
    download_embedding_model()