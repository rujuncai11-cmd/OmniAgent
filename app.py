"""
Streamlit RAG Web UI (app.py)
完整的 Web 应用，包括：
- 聊天界面
- 实时检索显示
- 参数调整
- 性能统计
- 对话历史保存

运行：streamlit run app.py
"""

import streamlit as st
import time
import os
from pathlib import Path
from datetime import datetime
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ======================== 配置 ========================
KNOWLEDGE_BASE_PATH = r"D:\HF_models\knowledge_base"
FAISS_INDEX_PATH = r"D:\HF_models\faiss_index"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = r"D:\HF_models"
CHAT_HISTORY_PATH = "chat_history.json"

# ======================== 页面配置 ========================
st.set_page_config(
    page_title="🤖 AI 研究助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .stChat {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #0066cc;
        margin-top: 0.5rem;
        border-radius: 0.25rem;
    }
    .stats-box {
        background-color: #fff4e6;
        padding: 0.8rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================== Session State 初始化 ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.vector_store = None
    st.session_state.total_tokens = 0

# ======================== 加载模型和向量库 ========================
@st.cache_resource
def load_models():
    """缓存加载模型"""
    with st.spinner("⏳ 加载模型中..."):
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        # 加载向量库
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cuda:0"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.vector_store = vector_store
        
        return tokenizer, model, vector_store

# ======================== RAG 检索函数 ========================
def retrieve_documents(query: str, top_k: int = 5):
    """检索相关文档"""
    retrieve_start = time.time()
    
    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(query, k=top_k)
    
    retrieve_time = time.time() - retrieve_start
    
    return docs, retrieve_time

# ======================== 生成答案函数 ========================
def generate_answer(query: str, documents, temperature: float, top_p: float):
    """生成答案"""
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    
    # 构建上下文
    context = "\n\n".join([
        f"【来源：{doc.metadata.get('filename', 'Unknown')}】\n{doc.page_content}"
        for doc in documents[:3]  # 只用前 3 篇
    ])
    
    # 构建 Prompt
    prompt = f"""你是一个 AI 研究助手，现在需要根据以下信息回答用户的问题。

【知识库信息】
{context}

【用户问题】
{query}

【回答要求】
- 直接、简洁地回答问题
- 基于知识库信息
- 如果知识库中没有相关信息，说"暂无相关信息"
- 长度：100-300 字

【答案】"""
    
    # 生成
    generate_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.2
    )
    
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    generate_time = time.time() - generate_start
    
    return answer.strip(), generate_time

# ======================== 加载对话历史 ========================
def load_chat_history():
    """加载对话历史"""
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# ======================== 保存对话历史 ========================
def save_chat_history():
    """保存对话历史"""
    with open(CHAT_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

# ======================== 主应用 ========================
def main():
    # 头部
    st.title("🤖 AI 研究助手")
    st.markdown("基于 RAG + Qwen 3B 的知识库问答系统")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 模型参数
        st.subheader("生成参数")
        temperature = st.slider(
            "温度 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="越低越确定，越高越随机"
        )
        
        top_p = st.slider(
            "概率阈值 (Top P)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="只考虑累积概率在此阈值内的 token"
        )
        
        top_k = st.slider(
            "检索文档数 (Top K)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="检索多少篇相关文档"
        )
        
        # 模型信息
        st.subheader("💻 系统信息")
        
        if st.button("🔄 初始化模型", key="load_btn"):
            load_models()
            st.success("✅ 模型加载成功！")
        
        if st.session_state.model is not None:
            st.write("✅ 模型已加载")
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.write(f"💾 显存: {allocated:.2f}GB / {total:.2f}GB")
        else:
            st.write("❌ 模型未加载")
        
        # 对话历史
        st.subheader("📝 对话历史")
        if st.button("🗑️ 清空对话", key="clear_btn"):
            st.session_state.messages = []
            save_chat_history()
            st.success("✅ 对话已清空")
        
        if st.button("💾 保存对话", key="save_btn"):
            save_chat_history()
            st.success("✅ 对话已保存")
        
        if st.button("📂 加载对话", key="load_history_btn"):
            history = load_chat_history()
            if history:
                st.session_state.messages = history
                st.success(f"✅ 加载了 {len(history)} 条消息")
            else:
                st.info("ℹ️ 没有保存的对话历史")
    
    # 主内容区
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 对话")
        
        # 显示对话历史
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander(f"📚 来源 ({len(message['sources'])} 篇)"):
                            for source in message["sources"]:
                                st.markdown(
                                    f"<div class='source-box'>"
                                    f"<strong>📄 {source['filename']}</strong><br>"
                                    f"{source['preview'][:200]}..."
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                    if "stats" in message:
                        with st.expander("⏱️ 性能统计"):
                            st.markdown(
                                f"<div class='stats-box'>"
                                f"📊 检索耗时: {message['stats']['retrieve_ms']:.0f}ms<br>"
                                f"🤖 生成耗时: {message['stats']['generate_ms']:.0f}ms<br>"
                                f"⏱️ 总耗时: {message['stats']['total_s']:.2f}s"
                                f"</div>",
                                unsafe_allow_html=True
                            )
    
    with col2:
        st.subheader("📊 统计信息")
        st.metric("对话数", len(st.session_state.messages))
        st.metric("总 tokens", st.session_state.total_tokens)
    
    # 输入框
    st.markdown("---")
    user_input = st.chat_input("输入你的问题...")
    
    if user_input:
        # 检查模型是否已加载
        if st.session_state.model is None:
            st.warning("⚠️ 请先在左侧点击 '初始化模型' 按钮")
        else:
            # 用户消息
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # 处理用户输入
            with st.spinner("🔍 检索中..."):
                total_start = time.time()
                
                # 检索
                docs, retrieve_time = retrieve_documents(user_input, top_k=top_k)
                
                # 生成
                answer, generate_time = generate_answer(
                    user_input, docs, temperature, top_p
                )
                
                total_time = time.time() - total_start
            
            # 助手消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "filename": doc.metadata.get('filename', 'Unknown'),
                        "preview": doc.page_content[:100]
                    }
                    for doc in docs
                ],
                "stats": {
                    "retrieve_ms": retrieve_time * 1000,
                    "generate_ms": generate_time * 1000,
                    "total_s": total_time
                }
            })
            
            # 更新统计
            st.session_state.total_tokens += len(
                st.session_state.tokenizer.encode(user_input + answer)
            )
            
            # 保存对话
            save_chat_history()
            
            # 刷新页面显示
            st.rerun()

if __name__ == "__main__":
    main()