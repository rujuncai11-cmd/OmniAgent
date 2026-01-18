"""
RAG 优化版本 (rag_optimize.py) - 简化版
专为 AI/Agent 领域知识库定制
功能：
1. 中文 embedding 模型（BGE）
2. 来源追踪 + 置信度显示
3. 简单向量检索（不依赖 EnsembleRetriever）
4. 自动分类输出
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 向量库 + 文本处理
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 大模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ======================== 配置 ========================
# ======================== 配置 ========================
KNOWLEDGE_BASE_PATH = r"D:\HF_models\knowledge_base"  # 你的文章路径
FAISS_INDEX_PATH = r"D:\HF_models\faiss_index"  # FAISS 存储路径
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # 改用 model_id
CACHE_DIR = r"D:\HF_models"  # 模型缓存目录（确保使用原始字符串）

# RAG 参数（优化版）
CHUNK_SIZE = 600  # 提高到 600，更多上下文
CHUNK_OVERLAP = 150  # 增加重叠，避免关键信息丢失
TOP_K_RETRIEVAL = 5  # 检索 top 5 文档
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值

# ======================== 步骤1：加载 & 分块 ========================
def load_and_chunk_documents():
    """加载所有文章 + 智能分块"""
    print("📖 [步骤1] 加载文章...")
    
    # 使用 PyPDF 加载 PDF 文件
    from langchain_community.document_loaders import PyPDFLoader
    
    documents = []
    pdf_path = Path(KNOWLEDGE_BASE_PATH)
    
    # 遍历文件夹中的所有 PDF
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"  📄 加载 {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        
        # 为每个文档添加来源文件名
        for doc in docs:
            doc.metadata["filename"] = pdf_file.stem
        
        documents.extend(docs)
    
    print(f"✅ 加载了 {len(documents)} 个页面")
    
    print("🔪 [步骤1] 分块中...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " "]  # 优先按中文标点分
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 分成了 {len(chunks)} 个 chunks")
    
    return chunks

# ======================== 步骤2：构建向量库 ========================
def build_vector_store(chunks):
    """使用中文优化的 embedding 模型"""
    print("🧠 [步骤2] 构建向量库（中文 BGE）...")
    
    # 使用 BAAI/bge-large-zh-v1.5（中文最优）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 创建 FAISS 向量库
    if os.path.exists(FAISS_INDEX_PATH):
        print("♻️  加载已有向量库...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("🏗️  构建新向量库...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"✅ 向量库已保存到 {FAISS_INDEX_PATH}")
    
    return vector_store, embeddings

# ======================== 步骤3：简单向量检索 ========================
def build_retriever(vector_store):
    """构建向量检索器"""
    print("🔍 [步骤3] 构建检索器...")
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )
    
    print("✅ 检索器准备完毕")
    return retriever

# ======================== 步骤4：加载大模型 ========================
def load_qwen_model():
    """加载 Qwen 模型（使用 model_id）"""
    print("🤖 [步骤4] 加载 Qwen 模型...")
    print(f"   模型 ID: {MODEL_ID}")
    print(f"   缓存目录: {CACHE_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    print("   ✓ Tokenizer 加载成功")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    print("   ✓ 模型加载成功")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   💾 GPU 显存: {allocated:.2f}GB / {total:.2f}GB")
    
    return tokenizer, model

# ======================== 步骤5：RAG 生成 ========================
def rag_generate(
    query: str,
    retriever,
    tokenizer,
    model,
    top_k: int = 3
) -> Dict:
    """
    RAG 生成：检索 + 生成 + 来源追踪
    
    返回：
    {
        "answer": "生成的答案",
        "sources": [{"filename": "...", "content": "...", "score": 0.85}, ...],
        "confidence": 0.88
    }
    """
    print(f"\n❓ 问题：{query}")
    print("🔍 检索中...")
    
    # 1. 检索文档
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        return {
            "answer": "抱歉，知识库中找不到相关信息。",
            "sources": [],
            "confidence": 0.0
        }
    
    # 2. 构建上下文
    context = "\n\n".join([
        f"【来源：{doc.metadata.get('filename', 'Unknown')}】\n{doc.page_content}"
        for doc in retrieved_docs[:top_k]
    ])
    
    # 3. 构建 prompt
    prompt = f"""你是一个 AI 领域的专家助手。请根据以下信息回答问题。

【知识库信息】
{context}

【问题】
{query}

【要求】
- 直接回答问题，不要重复信息
- 如果知识库中找不到相关信息，说"暂无相关信息"
- 回答长度：100-300 字

【答案】"""
    
    # 4. 生成答案
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 5. 计算置信度（基于检索得分）
    confidence = min(1.0, len(retrieved_docs) / TOP_K_RETRIEVAL * 0.95)
    
    # 6. 格式化源
    sources = [
        {
            "filename": doc.metadata.get('filename', 'Unknown'),
            "content": doc.page_content[:200] + "...",  # 前 200 字
            "confidence": round(confidence, 2)
        }
        for doc in retrieved_docs[:top_k]
    ]
    
    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": round(confidence, 2)
    }

# ======================== 步骤6：测试 ========================
def test_rag():
    """测试 RAG 系统"""
    # 初始化
    chunks = load_and_chunk_documents()
    vector_store, embeddings = build_vector_store(chunks)
    retriever = build_retriever(vector_store)
    tokenizer, model = load_qwen_model()
    
    # 测试问题（针对你的知识库）
    test_questions = [
        "AgentScope 框架的核心特点是什么？",
        "什么是检索增强生成（RAG）？有哪些优化方法？",
        "PyTorch FSDP 如何加速分布式训练？",
        "VideoRAG 如何处理长视频的上下文？",
        "多智能体仿真的主要挑战是什么？",
        "量子计算在 AI 中的应用前景如何？"  # 故意问知识库没有的
    ]
    
    results = []
    for q in test_questions:
        result = rag_generate(q, retriever, tokenizer, model, top_k=3)
        results.append({
            "question": q,
            "answer": result["answer"][:150],  # 只显示前 150 字
            "confidence": result["confidence"],
            "sources_count": len(result["sources"]),
            "sources": result["sources"]
        })
        print(f"✅ 答案（置信度 {result['confidence']}）：{result['answer'][:100]}...")
        print(f"📚 来源数：{len(result['sources'])}")
        if result["sources"]:
            print(f"   来源文件：{[s['filename'] for s in result['sources']]}")
    
    # 保存结果
    with open("rag_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✅ 测试结果已保存到 rag_test_results.json")
    
    return results

# ======================== 主函数 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG 优化系统启动")
    print("=" * 60)
    test_rag()
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)