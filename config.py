import os
from typing import Dict, Any, List

class Config:
    """配置参数类 - 优化版（移除Ollama）"""
    
    # DeepSeek配置（主模型）
    DEEPSEEK_API_KEY = "sk-c80fe2e104e84e48ad4882cf784e0f70"  # 你的API密钥
    DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek聊天模型
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    
    # 模型配置参数
    MODEL_CONFIG = {
        "temperature": 0.7,          # 创造性程度
        "top_p": 0.9,               # 核采样参数
        "max_tokens": 1024,         # 最大生成token数
        "frequency_penalty": 0.1,   # 频率惩罚
        "presence_penalty": 0.1,    # 存在惩罚
    }
    
    # 向量数据库配置
    VECTOR_DB_PATH = "./data/vector_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # 对话配置
    MAX_INTERACTION_COUNT = 5
    MAX_TOKEN_LIMIT = 4096
    KNOWLEDGE_DB_PATH = "./data/knowledge_db.json"
    
    # Agent配置
    AGENT_CONFIGS = {
        "research_assistant": {
            "name": "Research_Assistant",
            "system_message": "你是一个专业的科研流程专家，擅长蛋白质结构分析和生物信息学工作流程设计。请基于检索到的知识和用户需求，设计合理的工作流程。",
            "description": "主要负责科研流程的设计和优化"
        },
        "knowledge_retriever": {
            "name": "Knowledge_Retriever", 
            "system_message": "你负责从知识库中检索相关信息，提供给研究助手使用。",
            "description": "知识检索工具"
        },
        "workflow_validator": {
            "name": "Workflow_Validator",
            "system_message": "你负责验证工作流程的合理性和完整性，检查是否有遗漏步骤。",
            "description": "流程验证工具"
        },
        "deepseek_agent": {
            "name": "DeepSeek_Agent",
            "system_message": "你是一个专业的生物信息学专家，使用DeepSeek模型提供专业分析、回答用户问题并生成详细的工作流程。",
            "description": "使用DeepSeek模型进行核心分析和回答"
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