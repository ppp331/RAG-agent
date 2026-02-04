import json
import requests
from typing import List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import Config

class AgentManager:
    """智能体管理类 - 优化版（仅使用DeepSeek）"""
    
    def __init__(self, knowledge_base):
        self.config = Config()
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        self.interaction_count = 0
        
        # 初始化所有Agent
        self._init_autogen_agents()
    
    def _init_autogen_agents(self):
        """初始化AutoGen代理结构 - 优化版"""
        
        # DeepSeek API配置（共享配置）
        deepseek_config = {
            "config_list": [{
                "model": self.config.DEEPSEEK_MODEL,
                "api_key": self.config.DEEPSEEK_API_KEY,
                "base_url": self.config.DEEPSEEK_BASE_URL,
                "temperature": self.config.MODEL_CONFIG["temperature"],
                "max_tokens": self.config.MODEL_CONFIG["max_tokens"],
            }]
        }
        
        # 用户代理（处理用户输入）
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,  # 减少自动回复次数
            code_execution_config=False,
            system_message="你负责接收用户输入并协调其他代理的工作。"
        )
        
        # 研究助手（改为工具型Agent，不调用LLM）
        self.research_assistant = AssistantAgent(
            name=self.config.AGENT_CONFIGS["research_assistant"]["name"],
            system_message=self.config.AGENT_CONFIGS["research_assistant"]["system_message"],
            llm_config=False,  # 关键修改：不使用LLM
            description=self.config.AGENT_CONFIGS["research_assistant"]["description"]
        )
        
        # DeepSeek代理（核心LLM Agent）
        self.deepseek_agent = AssistantAgent(
            name=self.config.AGENT_CONFIGS["deepseek_agent"]["name"],
            system_message=self.config.AGENT_CONFIGS["deepseek_agent"]["system_message"],
            llm_config=deepseek_config,  # 使用共享配置
            description=self.config.AGENT_CONFIGS["deepseek_agent"]["description"]
        )
        
        # 知识检索工具
        self.knowledge_retriever = AssistantAgent(
            name=self.config.AGENT_CONFIGS["knowledge_retriever"]["name"],
            system_message=self.config.AGENT_CONFIGS["knowledge_retriever"]["system_message"],
            llm_config=False,
            description=self.config.AGENT_CONFIGS["knowledge_retriever"]["description"],
            function_map={
                "retrieve_knowledge": self.retrieve_knowledge_tool
            }
        )
        
        # 工作流程验证器
        self.workflow_validator = AssistantAgent(
            name=self.config.AGENT_CONFIGS["workflow_validator"]["name"],
            system_message=self.config.AGENT_CONFIGS["workflow_validator"]["system_message"],
            llm_config=False,
            description=self.config.AGENT_CONFIGS["workflow_validator"]["description"]
        )
        
        # 创建组聊天
        self.agents = [self.user_proxy, self.research_assistant, 
                      self.knowledge_retriever, self.workflow_validator,
                      self.deepseek_agent]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=8,  # 减少对话轮次，节约API调用
            speaker_selection_method="auto"  # 让manager自动选择
        )
        
        # GroupChatManager使用最小配置（只用于协调，不生成长内容）
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": [{
                    "model": self.config.DEEPSEEK_MODEL,
                    "api_key": self.config.DEEPSEEK_API_KEY,
                    "base_url": self.config.DEEPSEEK_BASE_URL,
                    "temperature": 0.3,  # 低创造性，专注于协调
                    "max_tokens": 100,   # 只生成协调指令
                }]
            }
        )
    
    def retrieve_knowledge_tool(self, query: str) -> str:
        """知识检索工具函数"""
        results = self.knowledge_base.retrieve_knowledge(query, top_k=3)
        if results:
            return "\n".join([f"• {item['content']} (相似度: {item['similarity']:.3f})" for item in results])
        return "未找到相关信息"
    
    def _call_deepseek(self, prompt: str) -> str:
        """调用DeepSeek API"""
        try:
            payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.MODEL_CONFIG["temperature"],
                "max_tokens": self.config.MODEL_CONFIG["max_tokens"],
                "top_p": self.config.MODEL_CONFIG["top_p"],
                "frequency_penalty": self.config.MODEL_CONFIG["frequency_penalty"],
                "presence_penalty": self.config.MODEL_CONFIG["presence_penalty"]
            }
            
            response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
                },
                json=payload,
                timeout=30  # 缩短超时时间
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API错误: {str(e)}"
    
    def generate_response(self, user_query: str) -> str:
        """生成回复 - 优化版"""
        if self.interaction_count >= self.config.MAX_INTERACTION_COUNT:
            return "已达到最大交互次数。请重新开始对话。"
        
        try:
            # 先检索相关知识
            knowledge_results = self.knowledge_base.retrieve_knowledge(user_query)
            knowledge_text = ""
            if knowledge_results:
                knowledge_text = "\n".join([f"相关知识点: {item['content']}" 
                                          for item in knowledge_results])
            
            # 使用autogen的组聊天
            self.user_proxy.initiate_chat(
                self.manager,
                message=f"{knowledge_text}\n\n用户问题: {user_query}",
                clear_history=False,
                max_turns=4  # 限制对话轮次
            )
            
            # 获取DeepSeek代理的回复
            response = ""
            for msg in reversed(self.group_chat.messages):
                if msg["name"] == self.config.AGENT_CONFIGS["deepseek_agent"]["name"]:
                    response = msg["content"]
                    break
            
            # 如果autogen没有返回有效回复，使用直接调用
            if not response or len(response.strip()) < 10:
                prompt = f"""基于以下知识：
{knowledge_text}

用户问题：{user_query}

请生成专业、详细的回答："""
                response = self._call_deepseek(prompt)
            
            # 更新对话历史
            self._update_conversation_history(user_query, response)
            
            return response
            
        except Exception as e:
            # 如果autogen失败，直接调用DeepSeek
            return self._fallback_response(user_query)
    
    def _fallback_response(self, user_query: str) -> str:
        """备用响应方案（直接调用DeepSeek）"""
        retrieved_knowledge = self.knowledge_base.retrieve_knowledge(user_query)
        knowledge_contents = [item["content"] for item in retrieved_knowledge]
        
        prompt = f"""基于以下知识：
{' '.join(knowledge_contents)}

用户问题：{user_query}

请生成专业、详细的回答："""
        
        response = self._call_deepseek(prompt)
        self._update_conversation_history(user_query, response)
        return response
    
    def _update_conversation_history(self, user_query: str, response: str):
        """更新对话历史"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.interaction_count += 1
        
        # 保持历史记录长度
        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_history = []
        self.interaction_count = 0
        self.group_chat.messages = []
    
    def get_conversation_stats(self) -> Dict:
        """获取对话统计"""
        return {
            "interaction_count": self.interaction_count,
            "max_interactions": self.config.MAX_INTERACTION_COUNT,
            "history_length": len(self.conversation_history),
            "model": self.config.DEEPSEEK_MODEL,
            "temperature": self.config.MODEL_CONFIG["temperature"],
            "active_agents": [agent.name for agent in self.agents]
        }
    
    def update_model_config(self, **kwargs):
        """动态更新模型配置"""
        for key, value in kwargs.items():
            if key in self.config.MODEL_CONFIG:
                self.config.MODEL_CONFIG[key] = value
                print(f"已更新 {key} 为 {value}")