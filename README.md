# OmniAgent
<<<<<<< HEAD
# OmniAgent 一个多模态AI智能体系统（正在开发中）
=======
一个多模态AI智能体系统（正在开发中）
# OmniAgent - 多模态智能助理系统

## 项目简介
（一句话描述：一个完全本地运行、可处理文本/图像/视频/音频的多模态AI Agent）

## 系统架构
多模块输入层输入（图像、文本、视频、音频）通过多模块编码器进行编码，传给LLM，运行ReAct机制，在Thought-Action-Observation循环中不断迭代，直到没有下一步Action后把latest的Observation进行相应模态的形式进行输出。

## 核心功能
- 多模态输入理解
- 智能任务规划（ReAct）
- 工具调用
- 多模态输出生成

## 技术栈
- LLM: Qwen2-0.5B
- Agent框架: LangGraph
- 多模态: CLIP + BLIP + Whisper + MoviePy
- 界面: Streamlit

## 开发进度
- [√] 阶段0: 环境搭建
- [x] 阶段1: 系统设计
...
>>>>>>> fb2331d (update README with architecture diagram)
