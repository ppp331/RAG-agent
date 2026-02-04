import json
import requests
import re
import time
from typing import List, Dict, Any, Generator
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import Config

class AgentManager:
    """智能体管理类 - 保留完整的AutoGen多智能体协作"""
    
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
        """初始化AutoGen代理结构 - 确保函数调用正常工作"""
        
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
            "cache_seed": None,
        }
        
        # 用户代理 - 启用函数调用
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            system_message="""你是用户代理，负责启动和管理多智能体协作流程。
当收到用户问题时，协调各个智能体共同完成任务。
当得到最终回答后，说"【流程完成】"来结束对话。""",
            function_map={
                "retrieve_knowledge": self.retrieve_knowledge_tool
            },
            is_termination_msg=lambda x: "【流程完成】" in x.get("content", "")
        )
        
        # 知识检索器 - 作为函数调用的一部分
        # 注意：我们使用用户代理来调用函数，知识检索器实际上只是函数
        self.knowledge_retriever = AssistantAgent(
            name="Knowledge_Retriever",
            system_message="""你负责协调知识检索。
当需要检索知识时，指导用户代理调用retrieve_knowledge函数。
函数会自动返回检索结果。""",
            llm_config=deepseek_config,
        )
        
        # 研究助手
        self.research_assistant = AssistantAgent(
            name="Research_Assistant",
            system_message="""你是一个专业的科研流程专家。
基于检索到的知识，设计详细的工作流程。
要求：结构清晰、步骤详细、包含具体工具和方法。
在回答最后加上"【设计完成】"。""",
            llm_config=deepseek_config,
            is_termination_msg=lambda x: "【设计完成】" in x.get("content", "")
        )
        
        # 工作流程验证器
        self.workflow_validator = AssistantAgent(
            name="Workflow_Validator",
            system_message="""你负责验证工作流程的完整性和合理性。
请检查并提供具体改进建议。
在回答最后加上"【验证完成】"。""",
            llm_config=deepseek_config,
            is_termination_msg=lambda x: "【验证完成】" in x.get("content", "")
        )
        
        # DeepSeek代理 - 最终整合
        self.deepseek_agent = AssistantAgent(
            name="DeepSeek_Agent",
            system_message="""你是最终整合专家。请基于所有讨论，生成最终的完整回答。
要求：整合所有有用信息，给出最专业、最完整的最终回答。
在回答最后明确加上"【最终回答】"。""",
            llm_config=deepseek_config,
            is_termination_msg=lambda x: "【最终回答】" in x.get("content", "")
        )
        
        # 创建组聊天
        self.agents = [
            self.user_proxy,
            self.knowledge_retriever,
            self.research_assistant,
            self.workflow_validator,
            self.deepseek_agent
        ]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
            send_introductions=True,
        )
        
        # GroupChat Manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": [base_config],
                "temperature": 0.3,
                "max_tokens": 300,
            },
            system_message="""你负责协调多智能体协作。
请按照以下顺序进行：
1. Knowledge_Retriever指导知识检索
2. Research_Assistant设计工作流程
3. Workflow_Validator验证流程
4. DeepSeek_Agent生成最终回答
5. User_Proxy结束流程

确保每个智能体完成自己的任务。""",
            human_input_mode="NEVER",
        )
    
    def retrieve_knowledge_tool(self, query: str) -> str:
        """知识检索工具函数"""
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
                response = "\n".join(response_lines)
                print(f"✅ 检索完成，找到 {len(results)} 条结果")
                return response
            else:
                print("⚠️ 未找到相关知识")
                return "⚠️ 未找到相关知识，请基于专业知识进行设计。"
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return f"检索失败: {str(e)}"
    
    def _typewriter_output(self, text: str, delay: float = 0.02) -> Generator[str, None, None]:
        """打字机效果输出生成器"""
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
        """执行AutoGen工作流程 - 改进版"""
        print(f"\n🚀 启动AutoGen多智能体协作流程...")
        
        # 清空历史消息
        if hasattr(self, 'group_chat'):
            self.group_chat.messages = []
        
        try:
            # 首先手动执行知识检索
            print(f"  1. 📚 知识检索中...")
            knowledge_result = self.retrieve_knowledge_tool(user_query)
            
            # 使用用户代理启动组聊天
            print(f"  2. 🤖 启动智能体协作...")
            
            # 构建完整的初始消息
            initial_message = f"""用户问题：{user_query}

我已经为您检索到了相关知识：
{knowledge_result}

请按照以下流程协作：
1. Research_Assistant基于检索到的知识设计详细工作流程
2. Workflow_Validator验证工作流程的完整性
3. DeepSeek_Agent整合所有信息生成最终回答
4. 完成后User_Proxy说"【流程完成】"

请开始协作。"""
            
            # 启动组聊天
            chat_result = self.user_proxy.initiate_chat(
                self.manager,
                message=initial_message,
                max_turns=8,  # 增加轮次
                summary_method="last_msg",
            )
            
            # 提取最终回答
            final_response = ""
            
            # 方法1：从聊天历史中提取DeepSeek_Agent的回答
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
            
            # 方法2：如果没找到，取最后一个非用户代理的消息
            if not final_response and hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict):
                        if msg.get("role") == "assistant" and msg.get("name") not in ["User_Proxy", "Knowledge_Retriever"]:
                            final_response = msg.get("content", "")
                            break
                    elif hasattr(msg, 'role') and msg.role == "assistant":
                        if hasattr(msg, 'name') and msg.name not in ["User_Proxy", "Knowledge_Retriever"]:
                            if hasattr(msg, 'content'):
                                final_response = msg.content
                                break
            
            # 方法3：使用总结
            if not final_response and hasattr(chat_result, 'summary'):
                final_response = chat_result.summary
            
            print(f"✅ AutoGen多智能体协作完成")
            
            # 清理标记
            if final_response:
                # 移除所有内部标记
                markers = ["【最终回答】", "【设计完成】", "【验证完成】", "【流程完成】"]
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
        """备用响应方案"""
        try:
            # 先检索知识
            print("使用备用方案...")
            knowledge_results = self.knowledge_base.retrieve_knowledge(user_query, top_k=3)
            knowledge_text = "【相关知识】\n"
            if knowledge_results:
                for item in knowledge_results:
                    knowledge_text += f"- {item['content'][:150]}\n"
            
            # 模拟多智能体流程
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
            
            # 模拟研究助手
            design_prompt = f"""{knowledge_text}

请作为研究助手，为以下问题设计详细的工作流程：
问题：{user_query}

要求：结构清晰、步骤详细、包含具体工具和方法。"""
            
            design_payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的科研流程专家。"},
                    {"role": "user", "content": design_prompt}
                ],
                "temperature": self.config.MODEL_CONFIG["temperature"],
                "max_tokens": 1500
            }
            
            design_response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=design_payload,
                timeout=30
            )
            
            if design_response.status_code != 200:
                return f"设计阶段API错误: {design_response.status_code}"
            
            design_content = design_response.json()["choices"][0]["message"]["content"]
            
            # 最终整合
            final_prompt = f"""用户问题：{user_query}

【相关知识】
{knowledge_text}

【流程设计】
{design_content}

请作为最终整合专家，基于以上信息生成最专业、最完整的最终回答。"""
            
            final_payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是最终整合专家，请给出最权威的完整回答。"},
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
                return f"最终整合API错误: {final_response.status_code}"
                
        except Exception as e:
            return f"备用方案失败: {str(e)}"
    
    def generate_response_with_typewriter(self, user_query: str) -> Generator[str, None, str]:
        """生成带有打字机效果的回复"""
        if self.interaction_count >= self.config.MAX_INTERACTION_COUNT:
            yield "已达到最大交互次数。请重新开始对话。\n"
            return "已达到最大交互次数"
        
        print(f"\n🔍 用户查询: {user_query}")
        
        # 使用AutoGen生成完整回答
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
        """生成普通回复（兼容）"""
        full_response = ""
        for chunk in self.generate_response_with_typewriter(user_query):
            full_response += chunk
        return full_response
    
    def _update_conversation_history(self, user_query: str, response: str):
        """更新对话历史"""
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
        """重置对话"""
        self.conversation_history = []
        self.interaction_count = 0
        if hasattr(self, 'group_chat'):
            self.group_chat.messages = []
        print("对话历史已重置")
    
    def update_model_config(self, **kwargs):
        """更新模型配置"""
        # 更新配置
        if 'temperature' in kwargs:
            self.config.MODEL_CONFIG['temperature'] = kwargs['temperature']
        if 'top_p' in kwargs:
            self.config.MODEL_CONFIG['top_p'] = kwargs['top_p']
    
    def get_conversation_stats(self) -> Dict:
        """获取对话统计"""
        return {
            "interaction_count": self.interaction_count,
            "max_interactions": self.config.MAX_INTERACTION_COUNT,
            "history_length": len(self.conversation_history),
            "model": self.config.DEEPSEEK_MODEL,
            "active_agents": [agent.name for agent in self.agents] if hasattr(self, 'agents') else [],
            "workflow_mode": "AutoGen多智能体协作",
            "output_mode": "打字机效果"
        }