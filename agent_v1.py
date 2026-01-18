"""
ReAct Agent 实现 (agent_v1.py)
完整的思考->行动->观察->结论循环

功能：
1. 思考（Thought）：分析问题需要什么信息
2. 行动（Action）：调用 RAG 工具检索知识库
3. 观察（Observation）：分析检索结果
4. 最终答案（Final Answer）：生成最终回答
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
import time
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ======================== 配置 ========================
KNOWLEDGE_BASE_PATH = r"D:\HF_models\knowledge_base"
FAISS_INDEX_PATH = r"D:\HF_models\faiss_index"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = r"D:\HF_models"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 3

# ======================== 工具1：RAG 检索工具 ========================
class RAGTool:
    """RAG 工具：从知识库检索信息"""
    
    def __init__(self):
        print("🛠️  [初始化] RAG 工具...")
        
        # 加载向量库
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cuda:0"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("   ✓ FAISS 向量库加载成功")
        else:
            raise FileNotFoundError(f"向量库不存在: {FAISS_INDEX_PATH}")
        
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
    
    def execute(self, query: str) -> Dict:
        """执行检索"""
        start_time = time.time()
        
        docs = self.retriever.invoke(query)
        
        if not docs:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "documents": [],
                "content": "知识库中找不到相关信息",
                "time_ms": elapsed * 1000
            }
        
        # 整理检索结果
        documents = [
            {
                "filename": doc.metadata.get('filename', 'Unknown'),
                "content": doc.page_content[:300],  # 前 300 字
                "full_content": doc.page_content
            }
            for doc in docs
        ]
        
        # 拼接所有文档内容
        all_content = "\n\n".join([
            f"【{doc['filename']}】\n{doc['full_content']}"
            for doc in documents
        ])
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "documents": documents,
            "content": all_content,
            "time_ms": elapsed * 1000
        }

# ======================== 工具2：计算器工具 ========================
class CalculatorTool:
    """简单计算器工具"""
    
    def execute(self, expression: str) -> Dict:
        """执行计算"""
        try:
            result = eval(expression)
            return {
                "success": True,
                "result": str(result)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ======================== ReAct Agent ========================
class ReActAgent:
    """ReAct Agent 实现"""
    
    def __init__(self, tokenizer, model):
        """初始化 Agent"""
        self.tokenizer = tokenizer
        self.model = model
        
        # 初始化工具
        self.tools = {
            "rag": RAGTool(),
            "calculator": CalculatorTool()
        }
        
        self.max_iterations = 5  # 最多迭代次数
        self.conversation_history = []  # 对话历史
    
    def parse_action(self, text: str) -> Tuple[str, str]:
        """解析 Action 和 Input"""
        # 匹配 Action: xxx 和 Action Input: yyy
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return action, action_input
        
        return None, None
    
    def execute_tool(self, tool_name: str, tool_input: str) -> Tuple[str, float]:
        """执行工具，返回结果和耗时"""
        start_time = time.time()
        
        if tool_name == "rag":
            result = self.tools["rag"].execute(tool_input)
            elapsed = time.time() - start_time
            
            if result["success"]:
                return f"检索到 {len(result['documents'])} 篇相关文档:\n{result['content'][:1000]}", elapsed
            else:
                return result["content"], elapsed
        
        elif tool_name == "calculator":
            result = self.tools["calculator"].execute(tool_input)
            elapsed = time.time() - start_time
            
            if result["success"]:
                return f"计算结果: {result['result']}", elapsed
            else:
                return f"计算失败: {result['error']}", elapsed
        
        else:
            elapsed = time.time() - start_time
            return f"未知工具: {tool_name}", elapsed
    
    def generate_response(self, user_query: str) -> str:
        """生成 Agent 回答（ReAct 循环）"""
        total_start = time.time()
        
        print("\n" + "="*70)
        print(f"👤 用户: {user_query}")
        print("="*70)
        
        thought_action_history = []
        time_stats = {
            "thinking": 0,
            "tool_execution": 0,
            "total": 0
        }
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            print(f"\n🔄 [迭代 {iteration + 1}/{self.max_iterations}]")
            
            # 1. 构建 Prompt（包含历史和指令）
            system_prompt = """你是一个 AI 研究助手。你能调用以下工具：

工具列表：
1. rag: 从知识库检索信息。格式: "rag(query)"
2. calculator: 执行计算。格式: "calculator(expression)"

使用以下格式回答问题：

Thought: 你对问题的思考（分析问题需要什么信息）
Action: 你要调用的工具名称
Action Input: 工具的输入参数
Observation: 工具的返回结果

...（重复思考-行动-观察直到得到最终答案）

Final Answer: 基于所有观察的最终回答"""

            # 构建对话
            prompt = f"""{system_prompt}

问题历史:
{chr(10).join(thought_action_history)}

当前问题: {user_query}

Thought:"""
            
            # 2. 模型生成思考和行动
            think_start = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # 减少到 100（从 200）
                temperature=0.5,     # 降低温度，更快收敛
                top_p=0.8,           # 更聚焦
                do_sample=True,
                repetition_penalty=1.2  # 防止重复
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            think_elapsed = time.time() - think_start
            time_stats["thinking"] += think_elapsed
            
            print(f"🤔 Thought:{response[:100]}...")
            print(f"   ⏱️  思考耗时: {think_elapsed*1000:.0f}ms")
            thought_action_history.append(f"Thought:{response}")
            
            # 3. 解析 Action
            action, action_input = self.parse_action(response)
            
            if action is None:
                # 没有找到 Action，可能是 Final Answer
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[-1].strip()
                    total_elapsed = time.time() - total_start
                    
                    print(f"\n✅ 最终答案:\n{final_answer}")
                    print(f"\n⏱️  性能统计:")
                    print(f"   │")
                    print(f"   ├─ 思考总耗时: {time_stats['thinking']*1000:.0f}ms")
                    print(f"   ├─ 工具总耗时: {time_stats['tool_execution']*1000:.0f}ms")
                    print(f"   ├─ 总耗时: {total_elapsed:.2f}s")
                    print(f"   └─ 迭代次数: {iteration + 1}")
                    
                    return final_answer
                else:
                    print(f"⚠️  无法解析 Action，重试...")
                    continue
            
            print(f"🔧 Action: {action}")
            print(f"📥 Input: {action_input}")
            
            # 4. 执行工具
            observation, tool_elapsed = self.execute_tool(action, action_input)
            time_stats["tool_execution"] += tool_elapsed
            
            print(f"👁️  Observation: {observation[:200]}...")
            print(f"   ⏱️  工具耗时: {tool_elapsed*1000:.0f}ms")
            
            thought_action_history.append(f"Action: {action}\nAction Input: {action_input}\nObservation: {observation}")
            
            iter_elapsed = time.time() - iter_start
            print(f"   ⏱️  迭代耗时: {iter_elapsed:.2f}s")
            
            # 5. 检查是否需要继续
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                total_elapsed = time.time() - total_start
                
                print(f"\n✅ 最终答案:\n{final_answer}")
                print(f"\n⏱️  性能统计:")
                print(f"   │")
                print(f"   ├─ 思考总耗时: {time_stats['thinking']*1000:.0f}ms")
                print(f"   ├─ 工具总耗时: {time_stats['tool_execution']*1000:.0f}ms")
                print(f"   ├─ 总耗时: {total_elapsed:.2f}s")
                print(f"   └─ 迭代次数: {iteration + 1}")
                
                return final_answer
        
        # 超过最大迭代次数
        total_elapsed = time.time() - total_start
        print(f"\n⚠️  达到最大迭代次数")
        print(f"\n⏱️  性能统计:")
        print(f"   │")
        print(f"   ├─ 思考总耗时: {time_stats['thinking']*1000:.0f}ms")
        print(f"   ├─ 工具总耗时: {time_stats['tool_execution']*1000:.0f}ms")
        print(f"   ├─ 总耗时: {total_elapsed:.2f}s")
        print(f"   └─ 迭代次数: {self.max_iterations}")
        
        return "抱歉，处理问题超时，请简化问题后重试。"

# ======================== 初始化 ========================
def load_qwen_model():
    """加载 Qwen 模型"""
    print("🤖 加载 Qwen 模型...")
    load_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    print(f"   ✓ Tokenizer 加载完成")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    print(f"   ✓ 模型加载完成")
    
    load_elapsed = time.time() - load_start
    print(f"   ⏱️  模型加载耗时: {load_elapsed:.2f}s")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   💾 显存: {allocated:.2f}GB / {total:.2f}GB")
    
    return tokenizer, model

# ======================== 主函数 ========================
def main():
    """主程序"""
    print("\n" + "="*70)
    print("🚀 ReAct Agent 系统启动")
    print("="*70)
    
    # 初始化
    tokenizer, model = load_qwen_model()
    agent = ReActAgent(tokenizer, model)
    
    print("\n✅ Agent 初始化完毕！")
    print("\n💡 输入你的问题（输入 'quit' 退出）:\n")
    
    # 交互循环
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() == 'quit':
                print("\n👋 再见！")
                break
            
            if not user_input:
                continue
            
            # Agent 生成回答
            response = agent.generate_response(user_input)
            print(f"\n🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 已中断")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()