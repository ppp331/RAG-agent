import json
import requests
import re
import time
from typing import List, Dict, Any, Generator
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import Config

class AgentManager:
    """智能体管理类 - 简化版，减少幻觉"""
    
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
    
    def _is_bioinformatics_query(self, query: str) -> bool:
        """判断是否与生信相关"""
        bio_keywords = [
            "蛋白质", "序列", "基因", "DNA", "RNA", "氨基酸", 
            "结构预测", "比对", "blast", "pdb", "结构",
            "protein", "sequence", "gene", "alignment", "蛋白质流程",
            "预测", "分析", "流程", "步骤", "方法", "工具"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in bio_keywords)
    
    def _init_autogen_agents(self):
        """初始化AutoGen代理结构 - 简化版"""
        
        # DeepSeek API配置
        base_config = {
            "model": self.config.DEEPSEEK_MODEL,
            "api_key": self.config.DEEPSEEK_API_KEY,
            "base_url": self.config.DEEPSEEK_BASE_URL,
            "api_type": "openai",
        }
        
        # 共享的llm配置 - 降低温度减少创造性
        deepseek_config = {
            "config_list": [base_config],
            "temperature": 0.1,  # 降低温度，减少幻觉
            "timeout": 120,
            "max_tokens": 800,  # 限制输出长度
        }
        
        # 用户代理
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            system_message="你是用户的代理，只负责提出用户的问题。不要自己回答问题。",
        )
        
        # 流程协调器 - 改为只做简单的转发，不生成内容
        self.workflow_coordinator = AssistantAgent(
            name="Workflow_Coordinator",
            system_message="""你是一个流程协调器。你的唯一任务是按顺序调用其他智能体，每次只说一句话：
1. 首先说："Knowledge_Retriever，请检索相关知识。"
2. 然后说："Research_Assistant，请整理流程。"
3. 然后说："Workflow_Validator，请验证完整性。"
4. 最后说："DeepSeek_Agent，请生成最终回答。"

不要添加任何其他内容，不要自己回答问题。""",
            llm_config=deepseek_config,
        )
        
        # 知识检索器 - 只返回原始知识
        self.knowledge_retriever = AssistantAgent(
            name="Knowledge_Retriever",
            system_message="""你是一个知识检索工具。你的任务：
1. 调用retrieve_knowledge工具获取相关知识
2. 只返回检索到的原始内容，不要添加任何解释
3. 如果没有找到相关知识，返回"【无相关知识】"
4. 完成后说"【检索完成】"

输出格式示例：
- 用户输入蛋白质序列 → 验证序列有效性 → 调用 API 预测结构 → 展示 3D 结构、氨基酸分布和 Ramachandran 图 → 提供 PDB 文件下载
- 序列有效性验证步骤：检查氨基酸字符是否有效，去除非法字符，验证序列长度
【检索完成】""",
            llm_config={
                "config_list": [base_config],
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
                "temperature": 0.1,
                "max_tokens": 500,
            },
            function_map={
                "retrieve_knowledge": self.retrieve_knowledge_tool
            },
        )
        
        # 研究助手 - 只整理格式，不添加内容
        self.research_assistant = AssistantAgent(
            name="Research_Assistant",
            system_message="""你是一个流程整理工具。你的任务：
1. 接收Knowledge_Retriever返回的原始知识
2. 只整理格式为Markdown列表，不添加任何新步骤
3. 如果有多条知识，按顺序列出
4. 完成后说"【整理完成】"

输出格式示例：
### 蛋白质结构预测流程
1. 验证序列有效性
   - 检查氨基酸字符是否有效
   - 去除非法字符
   - 验证序列长度
2. 调用API预测结构
   - 可使用AlphaFold2或RoseTTAFold
3. 展示结果
   - 3D结构可视化
   - 氨基酸分布图
   - Ramachandran图
4. 提供PDB文件下载
【整理完成】""",
            llm_config=deepseek_config,
        )
        
        # 工作流程验证器 - 只检查完整性
        self.workflow_validator = AssistantAgent(
            name="Workflow_Validator",
            system_message="""你是一个流程验证工具。你的任务：
1. 检查整理后的流程是否完整
2. 如果发现遗漏，只指出遗漏了什么
3. 不要生成新的流程内容
4. 如果完整，说"【验证通过】"，否则说"【验证失败：缺少XXX】"

输出格式示例：
【验证通过】
或
【验证失败：缺少"验证序列长度"步骤】""",
            llm_config=deepseek_config,
        )
        
        # DeepSeek代理 - 严格基于验证通过的流程
        self.deepseek_agent = AssistantAgent(
            name="DeepSeek_Agent",
            system_message="""你是一个回答生成工具。你的任务：
1. 接收验证通过的流程
2. 严格基于该流程生成最终回答
3. 可以添加必要的解释说明，但不能添加新步骤
4. 如果检索结果为"【无相关知识】"，回答"抱歉，我无法回答这个问题"
5. 如果验证失败，回答"抱歉，知识库中的流程不完整"

约束：
- 最终回答必须完全包含验证通过的流程中的所有步骤
- 不得添加流程中没有的步骤
- 回答要简洁清晰，直接列出步骤即可

输出格式示例：
根据知识库，蛋白质结构预测的流程如下：

1. **序列验证**：检查氨基酸字符有效性，去除非法字符，验证序列长度。
2. **结构预测**：使用AlphaFold2或RoseTTAFold API进行预测。
3. **结果展示**：生成3D结构可视化、氨基酸分布图和Ramachandran图。
4. **文件下载**：提供PDB文件下载链接。""",
            llm_config=deepseek_config,
        )
        
        # 创建组聊天 - 使用固定的发言顺序
        self.agents = [
            self.user_proxy,
            self.knowledge_retriever,  # 直接让知识检索器开始
            self.research_assistant,
            self.workflow_validator,
            self.deepseek_agent,
        ]
        
        # 不再使用GroupChatManager的自动选择，而是手动控制
        self.group_chat = None
        self.manager = None
    
    def retrieve_knowledge_tool(self, query: str) -> str:
        """知识检索工具 - 只返回原始内容"""
        try:
            # 先判断是否与生信相关
            if not self._is_bioinformatics_query(query):
                return "【无相关知识】\n【检索完成】"
            
            clean_query = self._clean_text(query)
            print(f"🔍 正在检索知识: {clean_query[:50]}...")
            results = self.knowledge_base.retrieve_knowledge(clean_query, top_k=3)
            
            if not results:
                print("⚠️ 未找到相关知识")
                return "【无相关知识】\n【检索完成】"
            
            # 只返回原始内容，不加任何修饰
            response_lines = []
            for item in results:
                content = self._clean_text(item['content'])
                response_lines.append(f"- {content}")
            
            response_lines.append("【检索完成】")
            print(f"✅ 检索完成，找到 {len(results)} 条结果")
            return "\n".join(response_lines)
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return f"检索失败\n【检索完成】"
    
    def _typewriter_output(self, text: str, delay: float = 0.02) -> Generator[str, None, None]:
        """打字机效果输出生成器"""
        if not text:
            return
        
        for char in text:
            yield char
            time.sleep(delay)
    
    def _execute_manual_workflow(self, user_query: str) -> str:
        """手动执行工作流程 - 避免AutoGen的自动发言问题"""
        print(f"\n🚀 启动手动控制的多智能体协作流程...")
        
        # 存储各步骤的结果
        knowledge_result = ""
        organize_result = ""
        validate_result = ""
        
        # 步骤1: 知识检索
        print("🤖 步骤1: Knowledge_Retriever 检索知识")
        knowledge_result = self.retrieve_knowledge_tool(user_query)
        print(f"   检索结果: {knowledge_result[:100]}...")
        
        # 如果无相关知识，直接返回
        if "【无相关知识】" in knowledge_result:
            return "抱歉，我无法回答这个问题（知识库中无相关信息）。"
        
        # 步骤2: 研究助手整理
        print("🤖 步骤2: Research_Assistant 整理流程")
        organize_result = self._call_research_assistant(knowledge_result)
        print(f"   整理完成")
        
        # 步骤3: 验证器验证
        print("🤖 步骤3: Workflow_Validator 验证完整性")
        validate_result = self._call_workflow_validator(organize_result)
        print(f"   验证结果: {validate_result}")
        
        # 步骤4: 生成最终回答
        print("🤖 步骤4: DeepSeek_Agent 生成回答")
        final_response = self._call_deepseek_agent(organize_result, validate_result)
        
        return final_response
    
    def _call_research_assistant(self, knowledge_text: str) -> str:
        """调用研究助手整理流程"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
            
            prompt = f"""你是一个流程整理工具。请将以下知识整理为Markdown列表格式：

{knowledge_text}

要求：
1. 只整理格式，不添加新步骤
2. 使用Markdown列表
3. 完成后说"【整理完成】"

整理结果："""
            
            payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个流程整理工具，只整理已有知识，不添加新内容。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return result
            else:
                return knowledge_text + "\n【整理完成】"
                
        except Exception as e:
            print(f"❌ 研究助手调用失败: {e}")
            return knowledge_text + "\n【整理完成】"
    
    def _call_workflow_validator(self, workflow_text: str) -> str:
        """调用验证器验证流程"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
            
            prompt = f"""你是一个流程验证工具。请检查以下流程是否完整：

{workflow_text}

如果完整，只说"【验证通过】"
如果不完整，说"【验证失败：缺少XXX】"
不要添加其他内容。"""
            
            payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个流程验证工具，只检查完整性。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return result
            else:
                return "【验证通过】"
                
        except Exception as e:
            print(f"❌ 验证器调用失败: {e}")
            return "【验证通过】"
    
    def _call_deepseek_agent(self, workflow_text: str, validation_result: str) -> str:
        """调用DeepSeek生成最终回答"""
        try:
            # 如果验证失败，返回错误信息
            if "【验证失败" in validation_result:
                return "抱歉，知识库中的流程不完整，无法生成回答。"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
            
            prompt = f"""你是一个回答生成工具。请基于以下流程生成最终回答：

{workflow_text}

要求：
1. 严格基于上述流程
2. 可以添加必要的解释说明
3. 不要添加新步骤
4. 回答要简洁清晰

最终回答："""
            
            payload = {
                "model": self.config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个回答生成工具，严格基于已有流程生成回答。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            response = requests.post(
                f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return result
            else:
                return "抱歉，生成回答时出现错误。"
                
        except Exception as e:
            print(f"❌ DeepSeek调用失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}"
    
    def generate_response_with_typewriter(self, user_query: str) -> Generator[str, None, str]:
        """生成带有打字机效果的回复"""
        # 首先判断是否与生信相关
        if not self._is_bioinformatics_query(user_query):
            sorry_msg = "抱歉，我只能回答生物信息学相关的问题（如蛋白质结构预测、序列分析等）。"
            for char in sorry_msg:
                yield char
                time.sleep(0.02)
            return sorry_msg
        
        if self.interaction_count >= self.config.MAX_INTERACTION_COUNT:
            sorry_msg = "已达到最大交互次数。请重新开始对话。"
            for char in sorry_msg:
                yield char
                time.sleep(0.02)
            return sorry_msg
        
        print(f"\n🔍 用户查询: {user_query}")
        
        # 使用手动控制的工作流程
        print("🤖 启动手动控制的多智能体协作...")
        final_response = self._execute_manual_workflow(user_query)
        
        # 检查回答
        if not final_response or len(final_response.strip()) < 20:
            # 再次尝试检索知识库
            results = self.knowledge_base.retrieve_knowledge(user_query, top_k=3)
            if results:
                final_response = "根据知识库，相关流程如下：\n\n"
                for i, item in enumerate(results, 1):
                    final_response += f"{i}. {item['content']}\n\n"
            else:
                final_response = "抱歉，知识库中没有找到相关的流程信息。"
        
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
        print("对话历史已重置")
    
    def update_model_config(self, **kwargs):
        """更新模型配置"""
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
            "active_agents": ["Knowledge_Retriever", "Research_Assistant", "Workflow_Validator", "DeepSeek_Agent"],
            "workflow_mode": "手动控制多智能体协作",
            "output_mode": "打字机效果"
        }