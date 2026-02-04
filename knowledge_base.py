import json
import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class KnowledgeBase:
    """知识库管理类"""
    
    def __init__(self):
        self.config = Config()
        
        # 先尝试加载本地模型，如果失败再下载
        self.embedding_model = self._load_embedding_model()
        
        self.knowledge_data = self._load_knowledge_db()
        self.vector_db = self._build_vector_db()
    
    def _load_embedding_model(self):
        """加载Embedding模型，支持本地缓存 - 使用国内镜像"""
        local_model_path = "./models/all-MiniLM-L6-v2"
        
        try:
            # 先尝试加载本地模型
            if os.path.exists(local_model_path):
                print("加载本地Embedding模型...")
                return SentenceTransformer(local_model_path)
            else:
                print("下载Embedding模型，使用国内镜像...")
                
                # 设置多个国内镜像源
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                
                # 使用国内镜像下载
                print("正在从镜像源下载，请耐心等待...")
                model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    use_auth_token=False,
                    cache_folder="./cache"
                )
                
                # 保存到本地
                os.makedirs(local_model_path, exist_ok=True)
                model.save(local_model_path)
                print("✅ 模型下载并保存完成")
                return model
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("尝试备用方案...")
            # 尝试使用预下载的模型
            return self._load_model_from_backup()
    
    def _load_model_from_backup(self):
        """备用方案：从预下载的模型文件加载"""
        import sys
        
        # 方法1：尝试使用 transformers 直接加载
        try:
            print("尝试直接加载模型...")
            from transformers import AutoModel, AutoTokenizer
            
            # 使用国内镜像
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"下载模型: {model_name}")
            
            # 下载到缓存
            cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                use_fast=True
            )
            model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(modules=[model])
            
            # 保存到本地
            local_path = "./models/all-MiniLM-L6-v2"
            os.makedirs(local_path, exist_ok=True)
            st_model.save(local_path)
            print("✅ 模型下载完成")
            return st_model
            
        except Exception as e:
            print(f"备用方案也失败: {e}")
            raise Exception(f"无法下载模型，请手动下载或检查网络连接。错误: {str(e)}")
    
    def _load_knowledge_db(self) -> List[Dict]:
        """加载知识库JSON数据"""
        if os.path.exists(self.config.KNOWLEDGE_DB_PATH):
            with open(self.config.KNOWLEDGE_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 初始化默认知识库
            default_knowledge = [
                {
                    "id": 1,
                    "type": "protein_workflow",
                    "tags": ["蛋白质", "结构预测", "3D可视化", "PDB"],
                    "content": "用户输入蛋白质序列（单条或多条）→ 验证序列有效性 → 调用 API 预测结构 → 展示 3D 结构、氨基酸分布和 Ramachandran 图 → 提供 PDB 文件下载",
                    "embedding": None
                },
                {
                    "id": 2,
                    "type": "fragment",
                    "tags": ["验证", "序列有效性"],
                    "content": "序列有效性验证步骤：检查氨基酸字符是否有效，去除非法字符，验证序列长度",
                    "embedding": None
                },
                {
                    "id": 3,
                    "type": "fragment", 
                    "tags": ["API调用", "结构预测"],
                    "content": "使用AlphaFold2或RoseTTAFold API进行蛋白质结构预测",
                    "embedding": None
                },
                {
                    "id": 4,
                    "type": "other_workflow",
                    "tags": ["基因分析", "序列比对"],
                    "content": "基因序列分析流程：输入DNA序列 → BLAST比对 → 基因注释 → 功能预测",
                    "embedding": None
                }
            ]
            self._save_knowledge_db(default_knowledge)
            return default_knowledge
    
    def _save_knowledge_db(self, data: List[Dict]):
        """保存知识库数据"""
        os.makedirs(os.path.dirname(self.config.KNOWLEDGE_DB_PATH), exist_ok=True)
        with open(self.config.KNOWLEDGE_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _build_vector_db(self):
        """构建向量数据库"""
        embeddings = []
        for item in self.knowledge_data:
            if item['embedding'] is None:
                text = f"{' '.join(item['tags'])} {item['content']}"
                embedding = self.embedding_model.encode(text)
                item['embedding'] = embedding.tolist()
            else:
                embedding = np.array(item['embedding'])
            embeddings.append(embedding)
        
        self._save_knowledge_db(self.knowledge_data)
        return np.array(embeddings)
    
    def add_knowledge(self, knowledge_type: str, tags: List[str], content: str):
        """添加新知识"""
        new_id = max([item['id'] for item in self.knowledge_data]) + 1
        new_item = {
            "id": new_id,
            "type": knowledge_type,
            "tags": tags,
            "content": content,
            "embedding": None
        }
        self.knowledge_data.append(new_item)
        self.vector_db = self._build_vector_db()
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关知识"""
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self.vector_db)[0]
        
        # 获取最相关的top_k个结果
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.knowledge_data[idx]["content"],
                "similarity": float(similarities[idx]),
                "type": self.knowledge_data[idx]["type"]
            })
        
        return results