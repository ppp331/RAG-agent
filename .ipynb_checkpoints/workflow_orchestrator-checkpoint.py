"""
工作流编排器 - 控制多智能体协作流程
"""
from typing import Dict, List, Any
import json

class WorkflowOrchestrator:
    """工作流编排器 - 管理多智能体协作"""
    
    def __init__(self, agents: Dict[str, Any], knowledge_base):
        self.agents = agents
        self.knowledge_base = knowledge_base
        self.conversation_log = []
        
    def execute_query_workflow(self, user_query: str) -> Dict:
        """执行查询工作流"""
        workflow_steps = [
            {
                "name": "knowledge_retrieval",
                "agent": self.agents["knowledge_retriever"],
                "task": f"检索关于'{user_query}'的信息",
                "description": "从知识库检索相关信息"
            },
            {
                "name": "expert_analysis", 
                "agent": self.agents["research_assistant"],
                "task": "基于检索信息设计详细工作流程",
                "description": "专家设计具体流程"
            },
            {
                "name": "quality_validation",
                "agent": self.agents["workflow_validator"],
                "task": "验证工作流程完整性",
                "description": "质量检查"
            },
            {
                "name": "final_synthesis",
                "agent": self.agents["deepseek_agent"],
                "task": "整合所有信息生成最终回答",
                "description": "最终整合输出"
            }
        ]
        
        results = {}
        context_accumulator = []
        
        print("🎯 开始执行多智能体工作流...")
        
        for step in workflow_steps:
            print(f"  🔄 步骤: {step['name']} - {step['description']}")
            
            # 构建当前步骤的上下文
            context = "\n".join(context_accumulator) if context_accumulator else ""
            
            # 执行步骤
            if step["name"] == "knowledge_retrieval":
                result = self._execute_knowledge_retrieval(user_query)
            else:
                result = self._execute_agent_step(
                    step["agent"], 
                    context, 
                    step["task"]
                )
            
            results[step["name"]] = result
            context_accumulator.append(f"【{step['description']}】\n{result}")
            
            print(f"    ✅ 完成")
        
        print("🎉 工作流执行完成")
        
        return {
            "final_answer": results["final_synthesis"],
            "intermediate_results": results,
            "workflow_steps": [step["name"] for step in workflow_steps]
        }
    
    def _execute_knowledge_retrieval(self, query: str) -> str:
        """执行知识检索"""
        results = self.knowledge_base.retrieve_knowledge(query, top_k=5)
        
        if not results:
            return "未找到相关知识"
        
        # 对结果进行简单分析
        relevant_results = [r for r in results if r["similarity"] > 0.3]
        irrelevant_results = [r for r in results if r["similarity"] <= 0.3]
        
        output = []
        output.append(f"找到 {len(results)} 条相关信息：")
        
        if relevant_results:
            output.append("\n【高相关度信息】")
            for i, r in enumerate(relevant_results[:3], 1):
                output.append(f"{i}. {r['content']} (相关度: {r['similarity']:.2f})")
        
        if irrelevant_results:
            output.append(f"\n【低相关度信息（{len(irrelevant_results)}条）】")
            output.append("这些信息相关性较低，供参考")
        
        return "\n".join(output)
    
    def _execute_agent_step(self, agent, context: str, task: str) -> str:
        """执行单个agent步骤"""
        # 这里简化实现，实际应该调用agent
        prompt = f"""上下文信息：
{context}

你的任务：{task}

请基于上下文完成任务："""
        
        # 模拟agent响应
        return f"完成: {task}"