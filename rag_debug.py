"""
RAG 诊断工具 (rag_debug.py)
- 查看知识库中的所有文档
- 看看"ai学习路径"的内容是否被正确加载
- 测试向量搜索结果
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_INDEX_PATH = r"D:\HF_models\faiss_index"
KNOWLEDGE_BASE_PATH = r"D:\HF_models\knowledge_base"

def diagnose():
    """诊断函数"""
    print("="*70)
    print("🔍 RAG 诊断工具")
    print("="*70)
    
    # 1. 检查知识库文件
    print("\n1️⃣  知识库文件列表:")
    print("-" * 70)
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    pdf_files = list(kb_path.glob("*.pdf"))
    
    print(f"总计：{len(pdf_files)} 个 PDF 文件\n")
    for i, pdf in enumerate(sorted(pdf_files), 1):
        size_mb = pdf.stat().st_size / 1e6
        print(f"{i:2}. 《{pdf.stem}》 ({size_mb:.1f}MB)")
    
    # 2. 检查是否有"学习路径"
    print("\n2️⃣  搜索 '学习路径' 相关文件:")
    print("-" * 70)
    learning_files = [f for f in pdf_files if "学习" in f.stem or "路径" in f.stem]
    if learning_files:
        for f in learning_files:
            print(f"✅ 找到: {f.stem}")
    else:
        print(f"❌ 未找到包含 '学习' 或 '路径' 的文件")
    
    # 3. 加载向量库并检查
    print("\n3️⃣  向量库统计:")
    print("-" * 70)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    try:
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ 向量库加载成功")
        print(f"   向量数: {vector_store.index.ntotal}")
    except Exception as e:
        print(f"❌ 向量库加载失败: {e}")
        return
    
    # 4. 测试搜索
    print("\n4️⃣  测试向量搜索:")
    print("-" * 70)
    
    test_queries = [
        "ai学习路径",
        "学习ai",
        "ai基础",
        "核心概念",
        "扫盲",
        "深度学习"
    ]
    
    for query in test_queries:
        print(f"\n🔎 搜索: '{query}'")
        docs = vector_store.similarity_search(query, k=5)
        
        for j, doc in enumerate(docs, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            content = doc.page_content[:60].replace('\n', ' ')
            print(f"   {j}. 《{filename}》")
            print(f"      {content}...")
    
    # 5. 检查特定文件是否在向量库中
    print("\n5️⃣  检查 'ai学习路径' 是否在向量库中:")
    print("-" * 70)
    
    # 通过搜索来验证
    docs = vector_store.similarity_search("ai学习路径 教程 入门 步骤", k=10)
    filenames = set([doc.metadata.get('filename', '') for doc in docs])
    
    print(f"搜索到的文件（去重）: {len(filenames)} 个")
    for fname in sorted(filenames):
        if fname:
            print(f"  - {fname}")
    
    if "ai学习路径" in str(filenames):
        print(f"\n✅ 'ai学习路径' 在向量库中")
    else:
        print(f"\n⚠️  'ai学习路径' 可能不在向量库中")
        print(f"   可能原因：")
        print(f"   1. 文件被删除或移动")
        print(f"   2. FAISS 索引是用旧的文件构建的")
        print(f"   3. 需要重新构建向量库")

if __name__ == "__main__":
    diagnose()