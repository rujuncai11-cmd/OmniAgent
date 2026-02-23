"""
Streamlit RAG Web UI - 第 1 周优化版（完整重写）
改进重点：
1. 极简提示词（适配 1.5B 小模型）
2. 低温度参数（完全确定性）
3. 混合检索 + 相似度过滤（过滤无关文档）
4. 严格截取回答部分，防止提示词泄露

运行: streamlit run app.py
"""

import streamlit as st
import time
import re
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ======================== 配置 ========================
KNOWLEDGE_BASE_PATH = r"D:\HF_models\knowledge_base"
FAISS_INDEX_PATH    = r"D:\HF_models\faiss_index"
MODEL_ID            = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR           = r"D:\HF_models"
CHAT_HISTORY_PATH   = "chat_history.json"

# 相似度过滤阈值（FAISS L2距离，越小越相关）
# 首次运行时看终端打印的分数，再来调整这个值
SIMILARITY_THRESHOLD = 1.0

# ======================== 页面配置 ========================
st.set_page_config(
    page_title="🤖 AI 论文研究助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #0066cc;
        margin-top: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .stats-box {
        background-color: #fff4e6;
        padding: 0.8rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #ffe8e8;
        padding: 0.8rem;
        border-left: 4px solid #ff6b6b;
        border-radius: 0.25rem;
    }
    .no-result-box {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================== Session State ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_questions": 0,
        "answered": 0,
        "unanswered": 0,
        "with_citation": 0
    }

# ======================== 加载模型 ========================
@st.cache_resource
def load_models():
    """加载模型和向量库，缓存避免重复加载"""
    try:
        st.info("⏳ 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        st.info("⏳ 加载大模型（Qwen 1.5B）...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        st.info("⏳ 加载向量模型和向量库...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return tokenizer, model, vector_store

    except Exception as e:
        st.error(f"❌ 加载失败: {str(e)}")
        return None, None, None

# ======================== 关键词提取 ========================
def extract_keywords(text: str) -> list:
    """提取中英文关键词，去除停用词"""
    stopwords = {
        '是', '什么', '如何', '为什么', '的', '了', '在', '有', '和',
        '以及', '等等', '请', '介绍', '说明', '解释', '描述', '告诉',
        'is', 'the', 'a', 'of', 'in', 'what', 'how', 'why', 'does'
    }
    # 中文：按字符提取，过滤停用词
    cn_chars = [w for w in text if '\u4e00' <= w <= '\u9fff' and w not in stopwords]
    # 英文：按空格分词，过滤停用词和短词
    en_words = [w for w in text.split() if len(w) > 2 and w.lower() not in stopwords]
    return cn_chars + en_words

# ======================== 混合检索 + 相似度过滤 ========================
def retrieve_documents_hybrid(query: str, top_k: int = 5):
    """
    混合检索：向量搜索 + 关键词匹配 + 相似度过滤
    
    关键改进：
    - 过滤 FAISS 距离 > SIMILARITY_THRESHOLD 的文档
    - 防止无关文档进入上下文导致幻觉
    """
    retrieve_start = time.time()
    vector_store = st.session_state.vector_store

    if vector_store is None:
        return [], 0.0

    try:
        # 1. 向量搜索（带分数）
        vector_docs_with_scores = vector_store.similarity_search_with_score(
            query, k=top_k * 2
        )

        # 调试：打印分数到终端（上线后可删）
        print(f"\n=== 检索分数调试 [查询: {query[:30]}] ===")
        for doc, score in vector_docs_with_scores:
            fname = doc.metadata.get('filename', '未知')
            print(f"  分数: {score:.4f} | 文件: {fname[:40]}")
        print(f"  当前阈值: {SIMILARITY_THRESHOLD}（低于此值才保留）")

        # 2. 相似度过滤（核心改进）
        filtered = [
            (doc, score) for doc, score in vector_docs_with_scores
            if score < SIMILARITY_THRESHOLD
        ]

        if not filtered:
            print("  ⚠️ 过滤后无文档，所有文档相似度不足！可尝试调高阈值。")
            retrieve_time = time.time() - retrieve_start
            return [], retrieve_time

        # 3. 关键词搜索（辅助补充精确匹配）
        keywords = extract_keywords(query)
        keyword_doc_set = set()
        keyword_extra = []

        if keywords:
            kw_docs = vector_store.similarity_search(query, k=top_k * 3)
            for doc in kw_docs:
                content_lower = doc.page_content.lower()
                for kw in keywords:
                    if kw.lower() in content_lower:
                        key = doc.page_content[:100]
                        if key not in keyword_doc_set:
                            keyword_doc_set.add(key)
                            keyword_extra.append(doc)
                        break

        # 4. 合并去重（向量过滤结果优先）
        doc_map = {}
        for doc, score in filtered:
            key = doc.page_content[:100]
            if key not in doc_map:
                doc_map[key] = (doc, 1.0 - score)  # 转为相似度（越高越好）

        for doc in keyword_extra:
            key = doc.page_content[:100]
            if key not in doc_map:
                doc_map[key] = (doc, 0.75)  # 关键词命中给固定分

        # 5. 按相似度排序，取 top_k
        sorted_docs = sorted(doc_map.values(), key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in sorted_docs[:top_k]]

        print(f"  ✅ 最终返回 {len(final_docs)} 篇文档")

    except Exception as e:
        st.error(f"检索失败: {e}")
        final_docs = []

    retrieve_time = time.time() - retrieve_start
    return final_docs, retrieve_time

# ======================== 构建提示词（极简版）========================
def build_prompt(query: str, documents: list) -> str:
    """
    极简提示词，专为 1.5B 小模型设计。
    
    原则：
    - 指令越短越好，小模型对长指令理解差
    - 用"只能""不能"等强制语气
    - 给模型一个清晰的输出起点【回答】
    """
    if documents:
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            filename = doc.metadata.get('filename', '未知论文')
            # 每篇截取 400 字，避免 context 太长
            content = doc.page_content[:400].strip()
            context_parts.append(f"文献{i}（{filename}）：\n{content}")
        context = "\n\n".join(context_parts)
    else:
        context = "（无相关文献）"

    prompt = f"""根据以下文献回答问题。只能使用文献中的信息，不能自己发挥或补充额外知识。

文献内容：
{context}

问题：{query}

要求：回答时引用文献名，如果文献中没有相关内容则说"文献中未提及"。

【回答】"""

    return prompt

# ======================== 生成答案 ========================
def generate_answer(query: str, documents: list, temperature: float, top_p: float):
    """
    调用 Qwen 1.5B 生成答案。
    
    关键优化：
    - do_sample=False：确定性输出，不随机
    - repetition_penalty=1.1：减少重复
    - 只解码新生成的 token，不含提示词
    - 截取【回答】后的内容，防止提示词泄露
    """
    tokenizer = st.session_state.tokenizer
    model     = st.session_state.model

    if model is None or tokenizer is None:
        return "❌ 模型未加载，请先点击左侧'初始化模型'", 0.0

    prompt = build_prompt(query, documents)

    try:
        generate_start = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temperature,
                top_p=top_p,
                do_sample=False,          # 确定性生成，不随机采样
                repetition_penalty=1.1,   # 抑制重复
                pad_token_id=tokenizer.eos_token_id,
            )

        # 只解码新生成的部分
        new_tokens = outputs[0][input_len:]
        raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 防止提示词泄露：截取【回答】之后的内容
        for marker in ["【回答】", "回答：", "答案："]:
            if marker in raw_answer:
                raw_answer = raw_answer.split(marker)[-1].strip()
                break

        generate_time = time.time() - generate_start
        return raw_answer, generate_time

    except Exception as e:
        return f"❌ 生成失败: {str(e)}", 0.0

# ======================== 辅助函数 ========================
def has_citation(answer: str) -> bool:
    """检查答案是否包含引用（文献名格式）"""
    pattern = r'文献\d+|（[^）]*[\u4e00-\u9fffA-Za-z][^）]*）|\([^)]*[\u4e00-\u9fffA-Za-z][^)]*\)'
    return bool(re.search(pattern, answer))

def is_no_answer(answer: str) -> bool:
    """检查是否是'无相关内容'类回答"""
    keywords = ["文献中未提及", "没有涉及", "无相关内容", "未找到", "知识库中未找到"]
    return any(kw in answer for kw in keywords)

# ======================== 主程序 ========================
def main():
    st.title("🤖 AI 论文研究助手")
    st.markdown("**第 1 周优化版** | 严谨的学术论文问答系统")

    # -------- 侧边栏 --------
    with st.sidebar:
        st.header("⚙️ 系统设置")

        # 生成参数
        st.subheader("🎛️ 生成参数")
        st.info("💡 低参数 = 更准确、更稳定")

        temperature = st.slider(
            "温度 (Temperature)",
            min_value=0.0, max_value=1.0,
            value=0.1, step=0.05,
            help="越低越准确。推荐 0.1"
        )
        top_p = st.slider(
            "概率阈值 (Top P)",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="越低越稳定。推荐 0.4-0.6"
        )
        top_k = st.slider(
            "检索文献数",
            min_value=1, max_value=10,
            value=5, step=1,
            help="检索几篇相关论文"
        )

        # 相似度阈值（高级）
        with st.expander("🔧 高级：相似度阈值"):
            threshold = st.slider(
                "过滤阈值（FAISS L2距离）",
                min_value=0.3, max_value=2.0,
                value=SIMILARITY_THRESHOLD, step=0.05,
                help="低于此值才保留。看终端打印分数来调整"
            )
            st.caption("分数越低说明越相关。调大保留更多，调小过滤更严。")
        # 注意：这里用局部变量 threshold 替代全局 SIMILARITY_THRESHOLD
        effective_threshold = threshold

        st.markdown("---")

        # 系统信息
        st.subheader("💻 系统信息")
        if st.button("🔄 初始化模型", key="load_btn"):
            with st.spinner("加载中...（首次需要 2-5 分钟）"):
                tokenizer, model, vector_store = load_models()
                if model is not None:
                    st.session_state.model        = model
                    st.session_state.tokenizer    = tokenizer
                    st.session_state.vector_store = vector_store
                    st.success("✅ 模型加载成功！")
                else:
                    st.error("❌ 加载失败，请检查路径配置")

        if st.session_state.model is not None:
            st.success("✅ 模型已加载")
        else:
            st.warning("⚠️ 模型未加载")

        st.markdown("---")

        # 统计信息
        st.subheader("📈 统计信息")
        total    = st.session_state.stats['total_questions']
        answered = st.session_state.stats['answered']
        citations = st.session_state.stats['with_citation']

        st.write(f"总提问数：{total}")
        st.write(f"有答案：{answered}")
        st.write(f"有引用：{citations}")

        if total > 0:
            st.metric("回答率", f"{answered / total * 100:.0f}%")
        if answered > 0:
            st.metric("引用率", f"{citations / answered * 100:.0f}%")

        st.markdown("---")

        # 操作按钮
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.stats = {
                "total_questions": 0,
                "answered": 0,
                "unanswered": 0,
                "with_citation": 0
            }
            st.session_state.total_tokens = 0
            st.success("✅ 已清空")

    # -------- 主内容区 --------
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("📊 统计")
        st.metric("总对话数", len(st.session_state.messages))
        st.metric("总 tokens", st.session_state.total_tokens)

    with col1:
        st.subheader("💬 对话")

        # 显示历史对话
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # 显示参考文献
                if message.get("sources"):
                    with st.expander(f"📚 参考文献（{len(message['sources'])} 篇）"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f"""<div class='source-box'>
                                <strong>📄 文献{i}：{source['filename']}</strong><br>
                                {source['preview'][:250]}...
                                </div>""",
                                unsafe_allow_html=True
                            )

                # 显示性能统计
                if message.get("perf"):
                    with st.expander("⏱️ 性能统计"):
                        p = message["perf"]
                        st.markdown(
                            f"""<div class='stats-box'>
                            🔍 检索：{p['retrieve_ms']:.0f}ms &nbsp;|&nbsp;
                            🤖 生成：{p['generate_ms']:.0f}ms &nbsp;|&nbsp;
                            ⏱️ 总计：{p['total_s']:.2f}s
                            </div>""",
                            unsafe_allow_html=True
                        )

    # -------- 输入框 --------
    user_input = st.chat_input("输入你的学术问题...")

    if user_input:
        # 检查模型是否加载
        if st.session_state.model is None:
            st.warning("⚠️ 请先点击左侧'初始化模型'按钮")
            st.stop()

        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.stats['total_questions'] += 1

        with st.chat_message("user"):
            st.write(user_input)

        # 处理问题
        with st.chat_message("assistant"):
            with st.spinner("🔍 检索文献中..."):
                total_start = time.time()

                # Step 1: 混合检索 + 相似度过滤
                # 临时覆盖全局阈值（用侧边栏的滑块值）
                import builtins
                _original = globals().get('SIMILARITY_THRESHOLD')
                globals()['SIMILARITY_THRESHOLD'] = effective_threshold

                docs, retrieve_time = retrieve_documents_hybrid(user_input, top_k=top_k)

                globals()['SIMILARITY_THRESHOLD'] = _original  # 恢复

                # Step 2: 如果没找到相关文档，直接返回提示
                if not docs:
                    answer = (
                        "📭 知识库中未找到与该问题相关的论文内容。\n\n"
                        "可尝试以下问题：\n"
                        "- LightRAG 如何实现检索增强生成？\n"
                        "- SAM 3D 的核心创新是什么？\n"
                        "- AgentScope 的多智能体架构是怎样的？\n"
                        "- VideoRAG 如何处理超长视频？\n"
                        "- Zep 的时序知识图谱是如何工作的？"
                    )
                    generate_time = 0.0
                    sources_meta  = []

                else:
                    # Step 3: 生成答案
                    with st.spinner("🤖 生成答案中..."):
                        answer, generate_time = generate_answer(
                            user_input, docs, temperature, top_p
                        )

                    sources_meta = [
                        {
                            "filename": doc.metadata.get('filename', '未知'),
                            "preview":  doc.page_content
                        }
                        for doc in docs
                    ]

                total_time = time.time() - total_start

            # 显示答案
            st.write(answer)

            # 显示来源
            if sources_meta:
                with st.expander(f"📚 参考文献（{len(sources_meta)} 篇）"):
                    for i, source in enumerate(sources_meta, 1):
                        st.markdown(
                            f"""<div class='source-box'>
                            <strong>📄 文献{i}：{source['filename']}</strong><br>
                            {source['preview'][:250]}...
                            </div>""",
                            unsafe_allow_html=True
                        )

            # 显示性能
            with st.expander("⏱️ 性能统计"):
                st.markdown(
                    f"""<div class='stats-box'>
                    🔍 检索：{retrieve_time * 1000:.0f}ms &nbsp;|&nbsp;
                    🤖 生成：{generate_time * 1000:.0f}ms &nbsp;|&nbsp;
                    ⏱️ 总计：{total_time:.2f}s &nbsp;|&nbsp;
                    📄 命中文献：{len(sources_meta)} 篇
                    </div>""",
                    unsafe_allow_html=True
                )

        # 更新统计
        has_ans = not is_no_answer(answer) and "未找到" not in answer
        has_cit = has_citation(answer)

        if has_ans:
            st.session_state.stats['answered'] += 1
        else:
            st.session_state.stats['unanswered'] += 1
        if has_cit:
            st.session_state.stats['with_citation'] += 1

        # 保存到历史
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources_meta,
            "perf": {
                "retrieve_ms": retrieve_time * 1000,
                "generate_ms": generate_time * 1000,
                "total_s":     total_time
            }
        })

        # 更新 token 统计
        try:
            st.session_state.total_tokens += len(
                st.session_state.tokenizer.encode(user_input + answer)
            )
        except Exception:
            pass

        st.rerun()


if __name__ == "__main__":
    main()