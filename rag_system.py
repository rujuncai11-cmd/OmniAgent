"""
✅ 完整的 RAG 系统（PDF + 3B 模型）- 修复版
- 正确的库导入路径
- 加载 PDF 文章
- 构建向量索引
- 基于 Qwen2.5-3B 的问答系统
- 必须返回出处
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

print("\n" + "="*70)
print("📚 RAG 系统初始化")
print("="*70)

# ============ Step 1: 安装和导入所需库 ============
print("\n1️⃣  导入库...")

# 先检查并安装必要的库
required_libs = {
    'PyPDF2': 'PyPDF2',
    'sentence_transformers': 'sentence-transformers',
    'faiss': 'faiss-cpu',
}

for lib_import, lib_install in required_libs.items():
    try:
        __import__(lib_import)
        print(f"   ✓ {lib_import}")
    except ImportError:
        print(f"   ⚠️  {lib_import} 未安装，安装中...")
        os.system(f"pip install {lib_install} -q")
        print(f"   ✓ {lib_import} 已安装")

# 导入库
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

print("   ✓ transformers")
print("   ✓ 所有库导入成功")

# ============ Step 2: 加载 Qwen2.5-3B 模型 ============
print("\n2️⃣  加载 Qwen2.5-3B 模型...")

model_id = "Qwen/Qwen2.5-3B-Instruct"
cache_dir = r"D:\HF_models"

try:
    print("   加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ tokenizer 加载成功")
    
    print("   加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ 模型加载成功")
    model.eval()
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    exit(1)

# ============ Step 3: PDF 加载和文本提取 ============
print("\n3️⃣  加载 PDF 文章...")

class PDFArticleLoader:
    """加载 PDF 文章并提取结构化信息"""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.documents = []
    
    def load_pdfs(self) -> List[Dict]:
        """加载所有 PDF 文件"""
        # 多种方式查找 PDF
        pdf_files = []
        
        # 方式 1: 递归查找
        try:
            pdf_files = list(Path(self.pdf_folder).glob("**/*.pdf"))
        except Exception as e:
            print(f"   警告：递归查找失败 ({e})，尝试直接查找...")
        
        # 方式 2: 直接在当前文件夹查找
        if not pdf_files:
            try:
                pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
            except:
                pass
        
        # 方式 3: 使用 os.listdir
        if not pdf_files:
            try:
                all_files = os.listdir(self.pdf_folder)
                pdf_files = [
                    Path(self.pdf_folder) / f 
                    for f in all_files 
                    if f.lower().endswith('.pdf')
                ]
            except Exception as e:
                print(f"   错误：无法访问目录 {self.pdf_folder}: {e}")
                return []
        
        if not pdf_files:
            print(f"   ⚠️  在 {self.pdf_folder} 中未找到 PDF 文件")
            # 打印目录内容用于调试
            try:
                contents = os.listdir(self.pdf_folder)
                print(f"   目录内容：{contents[:10]}")  # 显示前 10 个文件
            except:
                print(f"   无法读取目录内容")
            return []
        
        print(f"   找到 {len(pdf_files)} 个 PDF 文件")
        
        for pdf_file in pdf_files:
            try:
                print(f"   处理: {pdf_file.name}...", end="", flush=True)
                doc = self._extract_pdf_content(pdf_file)
                if doc:
                    self.documents.append(doc)
                    print(" ✓")
                else:
                    print(" ✗（文件为空）")
            except Exception as e:
                print(f" ✗ ({str(e)[:50]})")
        
        print(f"   总共加载 {len(self.documents)} 篇文章")
        return self.documents
    
    def _extract_pdf_content(self, pdf_path: Path) -> Dict:
        """提取 PDF 内容和元数据"""
        reader = PdfReader(pdf_path)
        
        # 提取文本
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # 尝试提取元数据
        metadata = reader.metadata or {}
        
        return {
            "filename": pdf_path.name,
            "text": text,
            "pages": len(reader.pages),
            "title": metadata.get("/Title", pdf_path.stem),
            "author": metadata.get("/Author", "未知"),
            "created_date": metadata.get("/CreationDate", "未知"),
            "keywords": metadata.get("/Keywords", ""),
        }

# 加载 PDF
pdf_loader = PDFArticleLoader(r"D:\HF_models\knowledge_base")
documents = pdf_loader.load_pdfs()

if not documents:
    print("\n   ✗ 没有加载到任何 PDF 文件！")
    print("   请检查 D:\\knowledge_base 文件夹中是否有 PDF 文件")
    exit(1)

# ============ Step 4: 文本分块（不用 langchain） ============
print("\n4️⃣  分块处理文本...")

class SimpleTextSplitter:
    """简单的文本分块器"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """按字符长度分块"""
        chunks = []
        overlap = self.chunk_overlap
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, docs: List[Dict]) -> List[Dict]:
        """分块并保留来源信息"""
        chunks = []
        
        for doc in docs:
            text_chunks = self.split_text(doc["text"])
            
            for i, chunk in enumerate(text_chunks):
                if len(chunk.strip()) > 100:  # 忽略太短的块
                    chunks.append({
                        "content": chunk,
                        "source_file": doc["filename"],
                        "source_title": doc["title"],
                        "source_author": doc["author"],
                        "source_date": doc["created_date"],
                        "source_keywords": doc["keywords"],
                        "chunk_id": i,
                    })
        
        return chunks

splitter = SimpleTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"   分块完成: {len(chunks)} 个文本块")

# ============ Step 5: 生成向量嵌入 ============
print("\n5️⃣  生成向量嵌入...")

print("   加载 embedding 模型...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("   ✓ embedding 模型加载成功")

print(f"   为 {len(chunks)} 个文本块生成向量...")
texts_to_embed = [chunk["content"] for chunk in chunks]
embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)
print(f"   ✓ 向量生成完成 (维度: {embeddings.shape[1]})")

# ============ Step 6: 构建 FAISS 索引 ============
print("\n6️⃣  构建 FAISS 向量索引...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
print(f"   ✓ 索引构建完成 (包含 {index.ntotal} 个向量)")

# ============ Step 7: RAG 检索和生成 ============
print("\n7️⃣  RAG 系统准备就绪！")
print("="*70)

def retrieve_and_generate(query: str, top_k: int = 1) -> Dict:
    """
    检索相关文章并生成回答（极速版）
    """
    
    print(f"\n📝 问题: {query}")
    print("-" * 70)
    
    # 1. 向量化查询
    query_embedding = embedding_model.encode([query])[0]
    
    # 2. 检索相关文本
    distances, indices = index.search(
        np.array([query_embedding]).astype(np.float32),
        min(top_k, len(chunks))
    )
    
    # 3. 组织上下文（只取前 500 字符）
    context = ""
    source_info = []
    
    for idx, distance in zip(indices[0], distances[0]):
        chunk = chunks[int(idx)]
        # 只取前 500 字符，避免上下文过长
        truncated_content = chunk['content'][:500]
        context += f"【{chunk['source_title']}】\n{truncated_content}\n\n"
        
        source_key = chunk['source_file']
        if source_key not in [s['file'] for s in source_info]:
            source_info.append({
                "file": source_key,
                "title": chunk['source_title'],
                "author": chunk['source_author'],
            })
    
    # 4. 极简提示词（关键！加快 50%）
    prompt = f"""参考：{context}

Q: {query}
A:"""
    
    # 5. 用 Qwen2.5-3B 生成回答
    print("🤖 模型生成中（预计 30-60 秒）...")
    start_time = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 进一步减小，加快 2 倍
            do_sample=False,  # 关闭采样，加快生成
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案部分
    if "A:" in answer:
        answer = answer.split("A:")[-1].strip()
    
    elapsed = time.time() - start_time
    
    return {
        "query": query,
        "answer": answer,
        "sources": source_info,
        "time": elapsed
    }

# ============ Step 8: 测试 ============
print("\n" + "="*70)
print("🧪 RAG 系统测试")
print("="*70)

test_queries = [
    "这些文章主要讲什么？",
    "文章中提到的关键概念有哪些？",
    "有哪些应用场景被提到？"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n\n{'='*70}")
    print(f"测试 {i}/3")
    print('='*70)
    
    result = retrieve_and_generate(query, top_k=2)
    
    print(f"\n💬 回答：")
    print(result["answer"])
    
    print(f"\n📚 来源文章：")
    for source in result["sources"]:
        print(f"   - {source['title']} (作者: {source['author']})")
    
    print(f"\n⏱️  耗时: {result['time']:.2f} 秒")

print("\n\n" + "="*70)
print("✅ RAG 系统测试完成！")
print("="*70)
print("\n💡 使用说明：")
print("   1. 修改 test_queries 列表来提出你自己的问题")
print("   2. 调整 top_k 参数来改变检索的相关文本数量")
print("   3. 修改 chunk_size 来调整分块大小")