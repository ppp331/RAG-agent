import os
from typing import Dict, Any, List

class Config:
    """配置参数类 - 优化版（使用百度API + DeepSeek）"""
    
    # DeepSeek配置（主LLM）
    DEEPSEEK_API_KEY = "sk-c80fe2e104e84e48ad4882cf784e0f70"  # 你的API密钥
    DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek聊天模型
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    
    # 百度文心千帆Embedding API配置
    BAIDU_API_KEY = "bce-v3/ALTAK-iT64za6LG6HJEhmTVtXjw/36d5a254fb9faf77f2bbba7bdf3178db0cf95048"
    BAIDU_SECRET_KEY = ""  # 百度可能需要secret_key，如果不需要就留空
    
    # Embedding配置
    EMBEDDING_API_PROVIDER = "baidu"  # 使用百度API
    EMBEDDING_MODEL = "embedding-v1"  # 百度文心千帆的embedding模型
    
    # 模型配置参数 - 降低温度减少幻觉
    MODEL_CONFIG = {
        "temperature": 0.1,          # 降低创造性，减少幻觉
        "top_p": 0.7,                # 降低多样性
        "max_tokens": 800,            # 减少最大token数
        "frequency_penalty": 0.0,     # 关闭频率惩罚
        "presence_penalty": 0.0,      # 关闭存在惩罚
    }
    
    # 向量数据库配置
    VECTOR_DB_PATH = "./data/vector_db"
    
    # 对话配置
    MAX_INTERACTION_COUNT = 5
    MAX_TOKEN_LIMIT = 2048  # 降低限制
    KNOWLEDGE_DB_PATH = "./data/knowledge_db.json"
    
    # Agent配置 - 简化版
    AGENT_CONFIGS = {
        "research_assistant": {
            "name": "Research_Assistant",
            "system_message": "你是一个流程整理工具，只整理已有知识，不添加新内容。",
            "description": "流程整理工具"
        },
        "knowledge_retriever": {
            "name": "Knowledge_Retriever", 
            "system_message": "你是一个知识检索工具，只返回原始知识。",
            "description": "知识检索工具"
        },
        "workflow_validator": {
            "name": "Workflow_Validator",
            "system_message": "你是一个流程验证工具，只检查完整性。",
            "description": "流程验证工具"
        },
        "deepseek_agent": {
            "name": "DeepSeek_Agent",
            "system_message": "你是一个回答生成工具，严格基于验证通过的流程生成回答。",
            "description": "回答生成工具"
        }
    }
    
    # 工具函数配置
    TOOL_CONFIGS = {
        "validate_sequence": {
            "name": "validate_protein_sequence",
            "description": "验证蛋白质序列有效性"
        },
        "format_workflow": {
            "name": "format_protein_workflow", 
            "description": "格式化蛋白质工作流程"
        }
    }