import json
from knowledge_base import KnowledgeBase
from agent_manager import AgentManager
from config import Config
from typing import List, Dict

class ResearchFlowAgent:
    """科研流程生成智能体主类 - 增强版"""
    
    def __init__(self):
        self.config = Config()
        self.knowledge_base = KnowledgeBase()
        self.agent_manager = AgentManager(self.knowledge_base)
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        return self.agent_manager.generate_response(user_query)
    
    def add_new_knowledge(self, knowledge_type: str, tags: List[str], content: str):
        """添加新知识到知识库"""
        self.knowledge_base.add_knowledge(knowledge_type, tags, content)
        print(f"已添加新知识: {content[:50]}...")
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return self.agent_manager.get_conversation_stats()
    
    def reset(self):
        """重置对话"""
        self.agent_manager.reset_conversation()
        print("对话已重置")
    
    def update_model_parameters(self, temperature: float = None, top_p: float = None):
        """更新模型参数"""
        updates = {}
        if temperature is not None:
            updates["temperature"] = temperature
        if top_p is not None:
            updates["top_p"] = top_p
        
        if updates:
            self.agent_manager.update_model_config(**updates)
            print("模型参数已更新")

# 使用示例
def main():
    # 初始化智能体
    agent = ResearchFlowAgent()
    
    print("科研流程生成智能体已启动！")
    print(f"模型: {Config.OLLAMA_MODEL}")
    print(f"Temperature: {Config.MODEL_CONFIG['temperature']}")
    print(f"最大交互次数: {Config.MAX_INTERACTION_COUNT}")
    print("命令: 'quit'退出, 'reset'重置, 'status'状态, 'param'更新参数")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                agent.reset()
                continue
            elif user_input.lower() == 'status':
                status = agent.get_status()
                print(f"状态: {status}")
                continue
            elif user_input.lower().startswith('param '):
                # 处理参数更新命令，如: "param temperature=0.8"
                parts = user_input.split()
                if len(parts) >= 2:
                    param_str = parts[1]
                    if '=' in param_str:
                        key, value = param_str.split('=')
                        try:
                            value = float(value)
                            if key == 'temperature':
                                agent.update_model_parameters(temperature=value)
                            elif key == 'top_p':
                                agent.update_model_parameters(top_p=value)
                        except ValueError:
                            print("参数值必须是数字")
                continue
            
            # 处理查询
            response = agent.process_query(user_input)
            print(f"\n助手: {response}\n")
            
            # 检查是否达到最大交互次数
            status = agent.get_status()
            if status['interaction_count'] >= status['max_interactions']:
                print("已达到最大交互次数，对话将重置...")
                agent.reset()
                
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    main()