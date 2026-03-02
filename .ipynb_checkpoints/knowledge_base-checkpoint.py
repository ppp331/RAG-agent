import json
import os
import numpy as np
from typing import List, Dict, Any
import requests
import time
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class KnowledgeBase:
    """知识库管理类 - 使用百度文心千帆API版本"""
    
    def __init__(self):
        self.config = Config()
        
        # 初始化百度API客户端
        self.embedding_api = BaiduEmbeddingAPI(
            self.config.BAIDU_API_KEY,
            self.config.BAIDU_SECRET_KEY
        )
        
        print("✅ 百度文心千帆Embedding API已初始化")
        
        self.knowledge_data = self._load_knowledge_db()
        self.vector_db = self._build_vector_db()
    
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
                    "content": "用户输入蛋白质序列（单条或多条）→ 验证序列有效性 → 调用 API 预测结构 → 展示 3D 结构、氨基酸分布和 Ramachandran 图 → 提供 PDB 文件下载"
                },
                {
                    "id": 2,
                    "type": "fragment",
                    "tags": ["验证", "序列有效性"],
                    "content": "序列有效性验证步骤：检查氨基酸字符是否有效，去除非法字符，验证序列长度"
                },
                {
                    "id": 3,
                    "type": "fragment", 
                    "tags": ["API调用", "结构预测"],
                    "content": "使用AlphaFold2或RoseTTAFold API进行蛋白质结构预测"
                },
                {
                    "id": 4,
                    "type": "other_workflow",
                    "tags": ["基因分析", "序列比对"],
                    "content": "基因序列分析流程：输入DNA序列 → BLAST比对 → 基因注释 → 功能预测"
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
        print("正在构建向量数据库...")
        embeddings = []
        
        for i, item in enumerate(self.knowledge_data):
            print(f"  处理知识 {i+1}/{len(self.knowledge_data)}...")
            
            text = f"{' '.join(item['tags'])} {item['content']}"
            
            try:
                embedding = self.embedding_api.get_embedding(text)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"  ❌ 获取知识 {i+1} 的嵌入失败: {e}")
                fallback_embedding = self.embedding_api._generate_fallback_embedding(text)
                embeddings.append(fallback_embedding)
        
        print("✅ 向量数据库构建完成")
        return np.array(embeddings)
    
    def add_knowledge(self, knowledge_type: str, tags: List[str], content: str):
        """添加新知识"""
        if len(self.knowledge_data) > 0:
            new_id = max([item['id'] for item in self.knowledge_data]) + 1
        else:
            new_id = 1
            
        new_item = {
            "id": new_id,
            "type": knowledge_type,
            "tags": tags,
            "content": content
        }
        
        self.knowledge_data.append(new_item)
        self._save_knowledge_db(self.knowledge_data)
        
        # 重新构建向量数据库
        self.vector_db = self._build_vector_db()
        
        print(f"✅ 已添加新知识: {content[:50]}...")
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关知识"""
        try:
            # 获取查询的嵌入向量
            query_embedding = self.embedding_api.get_embedding(query)
            
            if len(self.vector_db) == 0:
                print("⚠️  向量数据库为空")
                return []
            
            # 计算相似度
            similarities = cosine_similarity([query_embedding], self.vector_db)[0]
            
            # 设置相似度阈值，低于阈值的不返回
            threshold = 0.3
            valid_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
            
            if not valid_indices:
                print("⚠️  没有找到相似度足够的结果")
                return []
            
            # 获取最相关的top_k个结果
            valid_similarities = [(i, similarities[i]) for i in valid_indices]
            valid_similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in valid_similarities[:top_k]]
            
            results = []
            for idx in top_indices:
                results.append({
                    "content": self.knowledge_data[idx]["content"],
                    "similarity": float(similarities[idx]),
                    "type": self.knowledge_data[idx]["type"]
                })
            
            print(f"✅ 检索完成，找到 {len(results)} 条相关结果")
            return results
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            # 降级方案：关键词匹配
            return self._fallback_retrieval(query, top_k)
    
    def _fallback_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """降级检索方案（当API失败时）"""
        print("⚠️  使用降级检索方案（关键词匹配）")
        
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge_data:
            score = 0
            content = item["content"].lower()
            tags = [tag.lower() for tag in item["tags"]]
            
            # 简单关键词匹配
            for tag in tags:
                if tag in query_lower:
                    score += 1
            
            if any(word in content for word in query_lower.split()):
                score += 0.5
            
            if score > 0:
                results.append({
                    "content": item["content"],
                    "similarity": min(score / 3, 0.9),
                    "type": item["type"]
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class BaiduEmbeddingAPI:
    """百度文心千帆Embedding API客户端"""
    
    def __init__(self, api_key: str, secret_key: str = ""):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://aip.baidubce.com"
        
        self._parse_api_key()
        self.access_token = self._get_access_token()
        print(f"✅ 百度API初始化成功")
        self.embedding_cache = {}
        
    def _parse_api_key(self):
        """解析API Key格式"""
        parts = self.api_key.split('/')
        if len(parts) >= 3:
            self.ak = parts[-2]
            self.sk = parts[-1]
        else:
            self.ak = self.api_key
            self.sk = self.secret_key
    
    def _get_access_token(self):
        """获取百度访问令牌"""
        try:
            if self.api_key.startswith("bce-v3/"):
                print("使用新版百度API Key格式")
                return self.api_key
            
            token_url = f"{self.base_url}/oauth/2.0/token"
            
            params = {
                "grant_type": "client_credentials",
                "client_id": self.ak,
                "client_secret": self.sk
            }
            
            print(f"获取百度访问令牌...")
            response = requests.post(token_url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get("access_token", self.api_key)
                
        except Exception as e:
            print(f"⚠️  获取百度访问令牌失败，将使用降级模式: {e}")
            return self.api_key
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入向量"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = None
            
            if self.access_token.startswith("bce-v3/"):
                embedding = self._call_new_api(text)
            else:
                embedding = self._call_legacy_api(text)
            
            if embedding is not None:
                self.embedding_cache[text] = embedding
                return embedding
            else:
                raise Exception("API调用失败")
                
        except Exception as e:
            print(f"⚠️  API调用失败，使用降级嵌入方案: {e}")
            
            fallback_embedding = self._generate_fallback_embedding(text)
            self.embedding_cache[text] = fallback_embedding
            return fallback_embedding
    
    def _call_new_api(self, text: str):
        """调用新版百度API"""
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            
            payload = {
                "input": [text],
                "model": "embedding-v1"
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding_array = result["data"][0]["embedding"]
                    return np.array(embedding_array)
            
            print(f"API返回异常: {response.text[:100]}")
            return None
            
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def _call_legacy_api(self, text: str):
        """调用传统百度API"""
        try:
            url = f"{self.base_url}/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"
            
            params = {
                "access_token": self.access_token
            }
            
            payload = {
                "input": [text]
            }
            
            response = requests.post(url, params=params, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding_array = result["data"][0]["embedding"]
                    return np.array(embedding_array)
            
            return None
            
        except Exception as e:
            print(f"传统API调用失败: {e}")
            return None
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """生成降级嵌入向量"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        
        np.random.seed(seed)
        embedding = np.random.randn(384)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding