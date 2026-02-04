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
        """构建向量数据库 - 使用百度API生成嵌入"""
        print("正在构建向量数据库...")
        embeddings = []
        
        for i, item in enumerate(self.knowledge_data):
            print(f"  处理知识 {i+1}/{len(self.knowledge_data)}...")
            
            # 组合文本：标签 + 内容
            text = f"{' '.join(item['tags'])} {item['content']}"
            
            try:
                # 使用百度API获取嵌入向量
                embedding = self.embedding_api.get_embedding(text)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"  ❌ 获取知识 {i+1} 的嵌入失败: {e}")
                # 使用随机向量作为降级方案
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
            
            # 获取最相关的top_k个结果
            top_indices = similarities.argsort()[-top_k:][::-1]
            
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
                    "similarity": min(score / 3, 0.9),  # 归一化到0-0.9
                    "type": item["type"]
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class BaiduEmbeddingAPI:
    """百度文心千帆Embedding API客户端"""
    
    def __init__(self, api_key: str, secret_key: str = ""):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://aip.baidubce.com"
        
        # 解析API Key
        self._parse_api_key()
        
        # 尝试获取访问令牌
        self.access_token = self._get_access_token()
        print(f"✅ 百度API初始化成功")
        
        # 缓存已处理的文本，避免重复请求
        self.embedding_cache = {}
        
    def _parse_api_key(self):
        """解析API Key格式"""
        # 百度API Key格式：bce-v3/ALTAK-xxxx/xxxx
        parts = self.api_key.split('/')
        if len(parts) >= 3:
            self.ak = parts[-2]  # Access Key ID
            self.sk = parts[-1]  # Secret Access Key
        else:
            self.ak = self.api_key
            self.sk = self.secret_key
    
    def _get_access_token(self):
        """获取百度访问令牌"""
        try:
            # 新版百度API直接使用API Key
            if self.api_key.startswith("bce-v3/"):
                print("使用新版百度API Key格式")
                return self.api_key
            
            # 传统token获取（备用）
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
        # 检查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # 尝试两种API调用方式
            embedding = None
            
            if self.access_token.startswith("bce-v3/"):
                # 方法1：新版API
                embedding = self._call_new_api(text)
            else:
                # 方法2：传统API
                embedding = self._call_legacy_api(text)
            
            if embedding is not None:
                # 缓存结果
                self.embedding_cache[text] = embedding
                return embedding
            else:
                # 如果API调用失败，使用降级方案
                raise Exception("API调用失败")
                
        except Exception as e:
            print(f"⚠️  API调用失败，使用降级嵌入方案: {e}")
            
            # 生成确定性随机向量作为降级方案
            fallback_embedding = self._generate_fallback_embedding(text)
            
            # 缓存降级结果
            self.embedding_cache[text] = fallback_embedding
            return fallback_embedding
    
    def _call_new_api(self, text: str):
        """调用新版百度API"""
        try:
            # 百度文心千帆Embedding API端点
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
        """生成降级嵌入向量（确定性随机向量）"""
        # 使用文本哈希作为随机种子，确保相同文本生成相同向量
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 生成384维向量（百度Embedding的标准维度）
        embedding = np.random.randn(384)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class EnhancedKnowledgeBase(KnowledgeBase):
    """增强版知识库 - 支持混淆知识和专业扩展"""
    
    def __init__(self):
        super().__init__()
        self.knowledge_categories = {
            "expert_knowledge": [],  # 专业知识
            "confused_knowledge": [],  # 混淆知识
            "workflow_templates": [],  # 工作流模板
            "tool_recommendations": []  # 工具推荐
        }
    
    def add_expert_knowledge(self, domain: str, content: str, confidence: float = 1.0):
        """添加专业知识"""
        self.add_knowledge("expert_knowledge", [domain, "专业"], content)
    
    def add_confused_knowledge(self, misleading_content: str, correction: str):
        """添加混淆知识（用于测试鲁棒性）"""
        self.add_knowledge("confused_knowledge", ["混淆", "需验证"], 
                          f"错误信息: {misleading_content}\n正确信息: {correction}")
    
    def retrieve_with_filter(self, query: str, min_confidence: float = 0.3) -> List[Dict]:
        """带过滤的检索"""
        results = self.retrieve_knowledge(query)
        
        # 过滤低相关性结果
        filtered = [r for r in results if r["similarity"] >= min_confidence]
        
        # 按类型分组
        grouped = {}
        for r in filtered:
            k_type = r["type"]
            if k_type not in grouped:
                grouped[k_type] = []
            grouped[k_type].append(r)
        
        return {
            "high_confidence": [r for r in filtered if r["similarity"] > 0.7],
            "medium_confidence": [r for r in filtered if 0.4 <= r["similarity"] <= 0.7],
            "low_confidence": [r for r in filtered if r["similarity"] < 0.4],
            "by_type": grouped
        }