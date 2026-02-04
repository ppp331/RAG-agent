import json
from knowledge_base import KnowledgeBase
from agent_manager import AgentManager
from config import Config
from typing import List, Dict
import time
import re

class ResearchFlowAgent:
    """科研流程生成智能体主类"""
    
    def __init__(self):
        self.config = Config()
        self.knowledge_base = KnowledgeBase()
        self.agent_manager = AgentManager(self.knowledge_base)
    
    def _clean_input(self, text: str) -> str:
        """清理输入文本"""
        if not text:
            return ""
        # 移除控制字符和特殊字符
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = ''.join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
        return text.strip()
    
    def process_query(self, user_query: str, typewriter: bool = True, verbose: bool = False) -> str:
        """处理用户查询"""
        # 清理用户输入
        user_query = self._clean_input(user_query)
        
        if verbose:
            print("\n" + "="*60)
            print("🤖 多智能体工作流启动")
            print("-"*60)
        
        if typewriter:
            return self._process_with_typewriter(user_query, verbose)
        else:
            # 普通模式直接调用
            print("\n助手: ", end="", flush=True)
            response = self.agent_manager.generate_response(user_query)
            if verbose:
                print(f"\n📊 回答长度: {len(response)} 字符")
            return response
    
    def _process_with_typewriter(self, user_query: str, verbose: bool = False) -> str:
        """使用打字机效果处理查询"""
        start_time = time.time()
        
        full_response = ""
        
        # 使用打字机效果生成响应
        for chunk in self.agent_manager.generate_response_with_typewriter(user_query):
            full_response += chunk
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n\n📊 生成统计:")
            print(f"  ⏱️  耗时: {elapsed_time:.1f}秒")
            print(f"  📏 回答长度: {len(full_response)} 字符")
            
            stats = self.agent_manager.get_conversation_stats()
            print(f"  🤖 使用智能体: {len(stats['active_agents'])} 个")
            print(f"  🎯 工作流模式: {stats.get('workflow_mode', '标准')}")
            print(f"  ✨ 输出效果: {stats.get('output_mode', '普通')}")
        
        print("\n" + "="*60)
        
        return full_response
    
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

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                🧬 科研流程生成智能体 🧬                     ║
║                多智能体协作系统 v1.0                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# 使用示例
def main():
    # 打印启动横幅
    print_banner()
    
    # 初始化智能体
    print("正在初始化系统...")
    agent = ResearchFlowAgent()
    
    print("\n✅ 系统初始化完成！")
    print("=" * 60)
    print(f"📊 系统配置:")
    print(f"  主模型: {Config.DEEPSEEK_MODEL}")
    print(f"  Embedding: {Config.EMBEDDING_API_PROVIDER} ({Config.EMBEDDING_MODEL})")
    print(f"  Temperature: {Config.MODEL_CONFIG['temperature']}")
    print(f"  最大交互次数: {Config.MAX_INTERACTION_COUNT}")
    print("\n💡 可用命令:")
    print("  'quit' 或 'exit' - 退出程序")
    print("  'reset' - 重置对话历史")
    print("  'status' - 查看系统状态")
    print("  'param temperature=0.5' - 更新模型参数")
    print("  'param top_p=0.8' - 更新top_p参数")
    print("\n📝 提示: 回答将以打字机效果显示")
    print("=" * 60)
    
    # 测试系统状态
    print("\n正在测试系统连接...")
    try:
        status = agent.get_status()
        print(f"✅ 系统状态正常，已加载 {len(status.get('active_agents', []))} 个智能体")
    except Exception as e:
        print(f"⚠️  系统状态检查异常: {e}")
    
    print("\n" + "=" * 60)
    print("💬 开始对话 (直接输入问题，或输入命令):")
    
    conversation_count = 0
    
    while True:
        try:
            # 获取用户输入
            user_input = input(f"\n[第{conversation_count + 1}轮] 用户: ").strip()
            
            if not user_input:
                continue
                
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 感谢使用，再见！")
                break
            
            # 检查重置命令
            elif user_input.lower() == 'reset':
                agent.reset()
                conversation_count = 0
                print("✅ 对话已重置")
                continue
            
            # 检查状态命令
            elif user_input.lower() == 'status':
                status = agent.get_status()
                print(f"\n📊 系统状态:")
                print(f"  交互次数: {status['interaction_count']}/{status['max_interactions']}")
                print(f"  历史记录: {status['history_length']} 条")
                print(f"  工作模式: {status['workflow_mode']}")
                print(f"  输出模式: {status['output_mode']}")
                print(f"  活跃智能体: {', '.join(status['active_agents'])}")
                continue
            
            # 检查参数更新命令
            elif user_input.lower().startswith('param '):
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
                            else:
                                print(f"❌ 未知参数: {key}")
                        except ValueError:
                            print("❌ 参数值必须是数字")
                    else:
                        print("❌ 参数格式错误，请使用: param temperature=0.7")
                else:
                    print("❌ 参数命令格式错误")
                continue
            
            # 处理普通查询
            print("\n" + "=" * 60)
            print(f"🔍 处理查询: {user_input[:50]}...")
            
            # 设置详细模式：每3轮对话显示一次详细统计
            verbose = (conversation_count % 3 == 0)
            
            # 处理查询 - 使用打字机效果
            response = agent.process_query(user_input, typewriter=True, verbose=verbose)
            
            conversation_count += 1
            
            # 检查是否达到最大交互次数
            status = agent.get_status()
            if status['interaction_count'] >= status['max_interactions']:
                print("\n⚠️ 已达到最大交互次数，对话将自动重置...")
                agent.reset()
                conversation_count = 0
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到中断信号")
            confirm = input("确定要退出吗？(y/N): ").strip().lower()
            if confirm == 'y':
                print("\n👋 程序已终止")
                break
            else:
                print("继续运行...")
                continue
                
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 询问是否继续
            confirm = input("发生错误，是否继续？(y/N): ").strip().lower()
            if confirm != 'y':
                print("程序退出")
                break
            continue

if __name__ == "__main__":
    main()