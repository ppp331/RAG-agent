import json
import requests
import re
import time
from typing import List, Dict, Any, Generator
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import Config

class AgentManager:
    """智能体管理类 - 最小化修复，保留所有原有功能"""
    
    def __init__(self, knowledge_base):
        self.config = Config()
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        self.interaction_count = 0
        
        # 初始化所有Agent
        self._init_autogen_agents()
    
    def _clean_text(self, text: str) -> str:
        """清理文本中的无效字符"""
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = ''.join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
        return text
    
    def _init_autogen_agents(self):
        """初始化AutoGen代理结构 - 只修复配置格式，不改变架构"""
        
        # DeepSeek API配置
        base_config = {
            "model": self.config.DEEPSEEK_MODEL,
            "api_key": self.config.DEEPSEEK_API_KEY,
            "base_url": self.config.DEEPSEEK_BASE_URL,
            "api_type": "openai",
        }
        
        # 共享的llm配置
        deepseek_config = {
            "config_list": [base_config],
            "temperature": self.config.MODEL_CONFIG["temperature"],
            "timeout": 120,
            "max_tokens": self.config.MODEL_CONFIG["max_tokens"],
        }
        
        # 用户代理 - 保持原有逻辑
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            system_message="你是用户的代理，只负责提出用户的问题。不要自己回答问题。",
        )
        
        # 流程协调器 - 保持原有逻辑
        self.workflow_coordinator = AssistantAgent(
            name="Workflow_Coordinator",
            system_message="""你是流程协调器，负责管理多智能体协作流程。
你的职责：
1. 接收用户问题
2. 协调知识检索、流程设计、验证和最终整合
3. 确保流程按顺序进行
4. 在流程结束时说"【流程完成】"

具体步骤：
1. 首先指导知识检索
2. 然后指导流程设计
3. 接着指导流程验证
4. 最后指导最终整合
5. 流程完成后说"【流程完成】"并结束""",
            llm_config=deepseek_config,
        )
        
        # 知识检索器 - 关键修复：修正函数调用配置格式
        self.knowledge_retriever = AssistantAgent(
            name="Knowledge_Retriever",
            system_message="""你是知识检索专家。当流程协调器要求检索知识时：
1. 调用retrieve_knowledge工具函数获取相关知识
2. 分析检索结果
3. 提供知识总结
4. 完成后说"【知识检索完成】"以便流程继续""",
            llm_config={
                "config_list": [base_config],  # 只包含base_config
                "functions": [
                    {
                        "name": "retrieve_knowledge",
                        "description": "从知识库检索相关知识",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "检索查询"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ],
                # 修复：移除可能导致验证错误的参数
                "temperature": 0.1,
                "max_tokens": 500,
            },
            function_map={
                "retrieve_knowledge": self.retrieve_knowledge_tool
            },
        )
        
        # 研究助手 - 保持原有逻辑
        self.research_assistant = AssistantAgent(
            name="Research_Assistant",
            system_message="""你是科研流程设计专家。基于检索到的知识，设计详细的工作流程。
要求：
1. 结构清晰，使用Markdown格式
2. 步骤详细具体
3. 包含必要的工具和方法推荐
4. 完成后说"【设计完成】"以便流程继续""",
            llm_config=deepseek_config,
        )
        
        # 工作流程验证器 - 保持原有逻辑
        self.workflow_validator = AssistantAgent(
            name="Workflow_Validator",
            system_message="""你是流程验证专家。请检查工作流程的完整性和合理性。
检查要点：
1. 是否有遗漏的关键步骤
2. 逻辑顺序是否合理
3. 推荐工具是否合适
4. 技术细节是否正确
请提供具体的改进建议。
完成后说"【验证完成】"以便流程继续""",
            llm_config=deepseek_config,
        )
        
        # DeepSeek代理 - 保持原有逻辑
        self.deepseek_agent = AssistantAgent(
            name="DeepSeek_Agent",
            system_message="""你是最终整合专家。请基于所有讨论，生成最终的完整回答。
要求：
1. 整合所有有用信息
2. 给出最专业、最完整的最终回答
3. 结构清晰，使用Markdown格式
4. 包含具体步骤、工具、注意事项
5. 直接面向用户，不要提及内部讨论过程
6. 完成后说"【最终回答】"以便流程结束""",
            llm_config=deepseek_config,
        )
        
        # 创建组聊天 - 保持原有逻辑
        self.agents = [
            self.user_proxy,           # 代表用户提出问题
            self.workflow_coordinator, # 协调整个流程
            self.knowledge_retriever,  # 检索知识
            self.research_assistant,   # 设计流程
            self.workflow_validator,   # 验证流程
            self.deepseek_agent,       # 最终回答
        ]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=12,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
            send_introductions=False,
        )
        
        # GroupChat Manager - 保持原有逻辑
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": [base_config],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            system_message="""你是多智能体协作管理器。请协调智能体协作。

智能体及其职责：
1. User_Proxy：代表用户提出初始问题
2. Workflow_Coordinator：协调整个工作流程
3. Knowledge_Retriever：检索相关知识
4. Research_Assistant：设计工作流程
5. Workflow_Validator：验证流程完整性
6. DeepSeek_Agent：生成最终回答

请确保对话有序进行。""",
            human_input_mode="NEVER",
        )
    
    def retrieve_knowledge_tool(self, query: str) -> str:
        """知识检索工具函数 - 保持原有逻辑"""
        try:
            clean_query = self._clean_text(query)
            print(f"🔍 正在检索知识: {clean_query[:50]}...")
            results = self.knowledge_base.retrieve_knowledge(clean_query, top_k=3)
            
            if results:
                response_lines = ["✅ **检索到的相关知识**"]
                for i, item in enumerate(results, 1):
                    content = self._clean_text(item['content'])
                    similarity = item['similarity']
                    response_lines.append(f"{i}. {content} (相关度: {similarity:.2f})")
                response_lines.append("\n【知识检索完成】")
                response = "\n".join(response_lines)
                print(f"✅ 检索完成，找到 {len(results)} 条结果")
                return response
            else:
                response = "⚠️ 未找到相关知识，请基于专业知识进行设计。\n【知识检索完成】"
                print("⚠️ 未找到相关知识")
                return response
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return f"检索失败: {str(e)}\n【知识检索完成】"
    
    def _typewriter_output(self, text: str, delay: float = 0.02) -> Generator[str, None, None]:
        """打字机效果输出生成器 - 保持原有逻辑"""
        if not text:
            return
        
        # 按段落处理
        paragraphs = text.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                yield '\n\n'
                time.sleep(delay * 2)
                continue
            
            lines = paragraph.split('\n')
            for line_idx, line in enumerate(lines):
                if line.strip():
                    # 按字符输出
                    for char in line:
                        yield char
                        time.sleep(delay)
                else:
                    yield '\n'
                    time.sleep(delay * 1.5)
                
                if line_idx < len(lines) - 1:
                    yield '\n'
                    time.sleep(delay * 1)
            
            if para_idx < len(paragraphs) - 1:
                yield '\n\n'
                time.sleep(delay * 2.5)
    
    def _execute_autogen_workflow(self, user_query: str) -> str:
        """执行AutoGen工作流程 - 保持原有逻辑"""
        print(f"\n🚀 启动AutoGen多智能体协作流程...")
        
        # 清空历史消息
        if hasattr(self, 'group_chat'):
            self.group_chat.messages = []
        
        try:
            # User_Proxy发起对话
            print(f"🤖 启动智能体协作...")
            
            chat_result = self.user_proxy.initiate_chat(
                self.manager,
                message=user_query,  # User_Proxy传递用户的问题
                max_turns=12,
                summary_method="last_msg",
            )
            
            # 提取最终回答
            final_response = ""
            
            # 查找DeepSeek_Agent的最终回答
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict):
                        if msg.get("name") == "DeepSeek_Agent":
                            content = msg.get("content", "")
                            if content:
                                final_response = content
                                break
                    elif hasattr(msg, 'name') and msg.name == "DeepSeek_Agent":
                        if hasattr(msg, 'content'):
                            final_response = msg.content
                            break
            
            # 如果没找到，查找最后一个智能体的回答
            if not final_response and hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict):
                        if msg.get("role") == "assistant" and msg.get("name") != "User_Proxy":
                            content = msg.get("content", "")
                            if content:
                                final_response = content
                                break
                    elif hasattr(msg, 'role') and msg.role == "assistant":
                        if hasattr(msg, 'name') and msg.name != "User_Proxy":
                            if hasattr(msg, 'content'):
                                final_response = msg.content
                                break
            
            print(f"✅ AutoGen多智能体协作完成")
            
            # 清理标记
            if final_response:
                markers = ["【最终回答】", "【设计完成】", "【验证完成】", "【知识检索完成】", "【流程完成】"]
                for marker in markers:
                    final_response = final_response.replace(marker, "")
                final_response = final_response.strip()
            
            return final_response or "未能生成完整回答"
            
        except Exception as e:
            print(f"❌ AutoGen流程错误: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response(user_query)
    
    def _fallback_response(self, user_query: str) -> str:
        """备用响应方案 - 保持原有逻辑"""
        try:
            # 先检索知识
            print("使用备用方案...")
            knowledge_results = self.knowledge_base.retrieve_knowledge(user_query, top_k=3)
            knowledge_text = "【相关知识】\n"
            if knowledge_results:
                for item in knowledge_results:
                    knowledge_text += f"- {item['content'][:150]}\n"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
            
            final_prompt = f"""{knowledge_text}

请作为专业的生物信息学专家，详细回答以下问题：
问题：{user_query}

要求：
1. 结构清晰，使用Markdown格式
2. 步骤详细具体
3. 包含必要的工具和方法推荐
4. 给出完整的工作流程"""
            
            final_payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是专业的生物信息学专家。"},
                    {"role": "user", "content": final_prompt}
                ],
                "temperature": self.config.MODEL_CONFIG["temperature"],
                "max_tokens": 2000
            }
            
            final_response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=final_payload,
                timeout=30
            )
            
            if final_response.status_code == 200:
                return final_response.json()["choices"][0]["message"]["content"]
            else:
                return f"API错误: {final_response.status_code}"
                
        except Exception as e:
            return f"备用方案失败: {str(e)}"
    
    def generate_response_with_typewriter(self, user_query: str) -> Generator[str, None, str]:
        """生成带有打字机效果的回复 - 保持原有逻辑"""
        if self.interaction_count >= self.config.MAX_INTERACTION_COUNT:
            yield "已达到最大交互次数。请重新开始对话。\n"
            return "已达到最大交互次数"
        
        print(f"\n🔍 用户查询: {user_query}")
        
        # 使用AutoGen工作流程
        print("🤖 启动AutoGen多智能体协作...")
        final_response = self._execute_autogen_workflow(user_query)
        
        # 检查是否为空
        if not final_response or len(final_response.strip()) < 20:
            print("⚠️  回答过短，使用备用方案...")
            final_response = self._fallback_response(user_query)
        
        # 用打字机效果输出
        print("📝 输出回答: ")
        full_response = ""
        for chunk in self._typewriter_output(final_response):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk
        
        print()  # 换行
        
        # 更新历史
        self._update_conversation_history(user_query, full_response)
        
        return full_response
    
    def generate_response(self, user_query: str) -> str:
        """生成普通回复（兼容） - 保持原有逻辑"""
        full_response = ""
        for chunk in self.generate_response_with_typewriter(user_query):
            full_response += chunk
        return full_response
    
    def _update_conversation_history(self, user_query: str, response: str):
        """更新对话历史 - 保持原有逻辑"""
        clean_query = self._clean_text(user_query)
        clean_response = self._clean_text(response)
        
        self.conversation_history.append({
            "role": "user", 
            "content": clean_query,
            "timestamp": time.time()
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": clean_response,
            "timestamp": time.time()
        })
        self.interaction_count += 1
        
        # 限制历史长度
        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]
    
    def reset_conversation(self):
        """重置对话 - 保持原有逻辑"""
        self.conversation_history = []
        self.interaction_count = 0
        if hasattr(self, 'group_chat'):
            self.group_chat.messages = []
        print("对话历史已重置")
    
    def update_model_config(self, **kwargs):
        """更新模型配置 - 保持原有逻辑"""
        # 更新配置
        if 'temperature' in kwargs:
            self.config.MODEL_CONFIG['temperature'] = kwargs['temperature']
        if 'top_p' in kwargs:
            self.config.MODEL_CONFIG['top_p'] = kwargs['top_p']
    
    def get_conversation_stats(self) -> Dict:
        """获取对话统计 - 保持原有逻辑"""
        return {
            "interaction_count": self.interaction_count,
            "max_interactions": self.config.MAX_INTERACTION_COUNT,
            "history_length": len(self.conversation_history),
            "model": self.config.DEEPSEEK_MODEL,
            "active_agents": [agent.name for agent in self.agents] if hasattr(self, 'agents') else [],
            "workflow_mode": "AutoGen多智能体协作",
            "output_mode": "打字机效果"
        }