# 🤖 OmniAgent — AI 论文研究助手

> 基于 RAG + ReAct Agent 的学术论文智能问答系统，支持动态知识库管理与多工具自主规划。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 项目简介

OmniAgent 是一个面向 AI 学术研究场景的智能问答系统。用户可以上传任意 PDF 论文，系统自动解析并构建向量知识库，之后通过自然语言提问，Agent 会自主决策使用哪种检索策略，并基于论文内容给出有来源引用的严谨答案。

**核心亮点：**
- 无幻觉设计：相似度过滤机制确保只有相关文档进入上下文
- 动态知识库：无需重启，实时上传/删除论文，向量库自动更新
- 多工具 Agent：根据问题类型自动选择检索、对比或总结工具

---

## 🏗️ 系统架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────────┐
│              ReAct Agent                │
│                                         │
│  分析问题类型                            │
│      │                                  │
│      ├─── 含"对比/区别" ──► compare_papers  │
│      ├─── 含"介绍/总结" ──► summarize_paper │
│      └─── 其他问题     ──► search_papers   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           混合检索（Hybrid RAG）          │
│                                         │
│  向量检索（FAISS）                        │
│      +                                  │
│  关键词匹配                              │
│      +                                  │
│  相似度过滤（L2 距离阈值）                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           Qwen 2.5-1.5B-Instruct        │
│                                         │
│  极简提示词 + 确定性生成                  │
│  do_sample=False / temperature=0.1      │
└─────────────────────────────────────────┘
    │
    ▼
  带引用的答案输出
```

---

## ⚙️ 技术栈

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| 大语言模型 | Qwen2.5-1.5B-Instruct | 轻量级本地推理 |
| 向量模型 | BAAI/bge-large-zh-v1.5 | 中文语义向量化 |
| 向量数据库 | FAISS | 高效相似度检索 |
| 检索框架 | LangChain | RAG 流程编排 |
| Web 框架 | Streamlit | 交互界面 |
| PDF 解析 | PyPDFLoader | 文档加载与切分 |

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- CUDA（推荐，CPU 也可运行）
- 8GB+ 内存

### 安装

```bash
git clone https://github.com/rujuncai11-cmd/omniagent.git
cd omniagent
pip install -r requirements.txt
```

### 配置路径

修改 `app.py` 顶部的配置项：

```python
KNOWLEDGE_BASE_PATH = r"你的路径\knowledge_base"   # PDF 存放目录
FAISS_INDEX_PATH    = r"你的路径\faiss_index"       # 向量库目录
CACHE_DIR           = r"你的路径\models"            # 模型缓存目录
```

### 运行

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，点击左侧「初始化模型」按钮加载模型（首次约 3-5 分钟）。

---

## ✨ 功能说明

### 1. 智能问答（RAG）

基于知识库内容回答问题，每个答案都附带来源引用。

```
用户：LightRAG 和传统 RAG 有什么区别？
系统：LightRAG 采用双层检索策略... (来源：LightRAG论文.pdf)
```

### 2. ReAct Agent 多工具路由

Agent 自动识别问题类型并调用对应工具：

| 触发词 | 工具 | 适用场景 |
|--------|------|----------|
| 区别、对比、比较、vs | compare_papers | 多篇论文横向对比 |
| 介绍、总结、概述 | summarize_paper | 单篇论文结构化总结 |
| 其他问题 | search_papers | 通用检索问答 |

### 3. 动态知识库管理

无需重启服务，实时管理论文：

- **上传**：拖入 PDF → 自动解析 → 切分 → 向量化 → 合并进知识库
- **删除**：点击删除按钮 → 自动重建向量库

---

## 🔧 核心技术实现

### 相似度过滤（防幻觉）

普通 RAG 不管相关性如何，把所有检索结果塞入上下文，导致小模型产生幻觉。本项目用 FAISS L2 距离做阈值过滤：

```python
docs_with_scores = vector_store.similarity_search_with_score(query, k=10)

# L2 距离越小越相关，超过阈值直接丢弃
filtered = [(doc, s) for doc, s in docs_with_scores if s < SIMILARITY_THRESHOLD]

if not filtered:
    return []   # 没有相关文档，不调用模型，从源头杜绝幻觉
```

### 混合检索（Hybrid Search）

向量检索擅长语义理解，关键词检索擅长精确匹配，两路融合：

```python
# 向量检索（主力）
vector_docs = vector_store.similarity_search_with_score(query, k=10)

# 关键词检索（补充）
for doc in all_docs:
    if any(kw in doc.page_content for kw in keywords):
        keyword_extras.append(doc)

# 合并去重，按相关度排序
merged = sorted(doc_map.values(), key=lambda x: x[1], reverse=True)
```

### Rule-Based Agent 路由

1.5B 小模型理解复杂指令能力弱，用规则匹配代替 LLM 做工具路由，稳定性更高：

```python
def agent_decide_tool(query):
    if any(kw in query for kw in ['区别', '对比', 'vs']):
        return "compare_papers", query
    if any(kw in query for kw in ['介绍', '总结']):
        return "summarize_paper", query
    return "search_papers", query
```

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 知识库论文数 | 12 篇 |
| 文档块总数 | 2,312 个 |
| 回答率 | 100% |
| 引用率 | 100% |
| 本地响应时间 | 3-18s |
| 运行设备 | CUDA GPU |

---

## 🪲 踩坑记录

**1. 1.5B 模型严重幻觉**

现象：模型无视检索上下文，用自身训练数据回答，出现 Docker、文件系统等无关内容。

解决：极简化提示词（从复杂多规则压缩到5行）+ `do_sample=False` 关闭随机采样 + 相似度过滤切断无关上下文。

**2. 检索到无关文档**

现象：FAISS 不管相关性，凑够 k 篇返回，低质量文档污染上下文。

解决：改用 `similarity_search_with_score`，基于 L2 距离设阈值，超过阈值不进上下文。

**3. 7 篇 PDF 文本提取失败**

现象：知乎打印版 PDF 是图片格式，PyPDFLoader 无法提取文字，返回空内容。

解决：诊断脚本逐一检测，删除问题 PDF，保留 12 篇可用论文。

**4. 提示词内容泄露到答案**

现象：小模型把提示词里的 `【回答】` 等标记原样输出。

解决：对输出做截取处理，只保留标记之后的内容。

**5. Agent 路由不稳定**

现象：让小模型自己决定用哪个工具，理解偏差导致工具选错。

解决：改用 rule-based 路由，关键词正则匹配决定工具，牺牲灵活性换稳定性。

---

## 📁 项目结构

```
omniagent/
├── app.py                  # 主程序（RAG + Agent + UI）
├── requirements.txt        # 依赖列表
├── build_vector_store.py   # 向量库初始构建脚本
├── test_vector_store.py    # 向量库诊断脚本
├── test_all_papers.py      # 论文质量检测脚本
├── chat_history.json       # 对话历史
└── .streamlit/
    └── config.toml         # Streamlit 配置
```

---

## 🗺️ 后续计划

- [ ] 接入 Deepseek API 替换本地小模型，提升回答质量
- [ ] 添加对话记忆，支持多轮上下文理解
- [ ] OCR 支持，处理扫描版 PDF
- [ ] 评估体系，量化准确率变化

---

## 📄 License

MIT License
