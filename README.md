# Agentic RAG System

基于 **LangChain + LangGraph + Google Gemini** 的智能检索增强生成系统。

不同于传统的线性 RAG 流水线（Retrieve → Augment → Generate），本项目通过 LangGraph 状态机实现了 **Agentic RAG**——让 LLM 作为推理引擎自主决策检索策略、评估文档质量、重写查询并验证回答。

## 核心能力

| 能力 | 说明 | 对应模式 |
|------|------|---------|
| **查询路由** | 根据问题类型智能选择检索策略（5 条路径） | Adaptive RAG |
| **Multi-Query** | 从多角度生成子查询，合并去重提升召回率 | Query Transformation |
| **问题拆解** | 将复杂问题分解为子问题逐步回答再合成 | Query Decomposition |
| **文档评分** | LLM 评估检索文档相关性，过滤无关内容 | Corrective RAG |
| **查询重写** | 检索失败时自动优化查询（最多 3 次） | Corrective RAG |
| **幻觉检测** | 检查回答是否基于检索到的事实 | Self-RAG |
| **引用溯源** | 回答中标注来源文件和位置 | Citation |
| **对话记忆** | 多轮对话上下文保持 | Memory |
| **Web 搜索** | 知识库外问题自动转 Web 搜索 | Tool Use |

## 技术栈

- **LLM**: Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai`
- **Embeddings**: `gemini-embedding-001` via `GoogleGenerativeAIEmbeddings`
- **向量数据库**: ChromaDB（本地持久化）
- **编排框架**: LangGraph（StateGraph + MemorySaver）
- **前端**: Streamlit Chat UI
- **Web 搜索**: Tavily（可选）

## 架构流程

```
用户输入
   ↓
┌──────────┐
│ 查询路由  │── 简单检索 ──→ 向量检索 ──→ 文档评分 ──→ 生成回答 ──→ 答案评估 ──→ 输出
│          │── 多角度  ──→ Multi-Query ──→ 文档评分 ──↗
│          │── 复杂问题 ──→ 问题拆解 ──→ 逐步回答 ──→ 合成 ──→ 答案评估 ──→ 输出
│          │── Web搜索 ──→ Tavily ──→ 生成回答 ──↗
│          │── 直接回答 ──→ 生成回答 ──↗
└──────────┘
                        ↑
              文档评分不通过 → 查询重写 → 重新检索（最多 3 次）
```

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/WjrRyan/LangchainRAG.git
cd LangchainRAG

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 Google API Key
```

你需要一个 Google AI Studio API Key：https://aistudio.google.com/apikey

（可选）Tavily API Key 用于 Web 搜索：https://tavily.com

### 3. 文档摄入

```bash
# 摄入单个文件
python ingest.py path/to/document.pdf

# 摄入目录下所有支持的文件
python ingest.py --dir path/to/documents/

# 查看向量库状态
python ingest.py --stats

# 清空向量库
python ingest.py --clear
```

### 4. 启动 Web UI

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501，即可：
- 左侧上传文档（PDF / Markdown / CSV）
- 主界面进行对话问答
- 查看 Agent 推理过程和引用来源

## 项目结构

```
LangchainRAG/
├── app.py                  # Streamlit Web UI 入口
├── ingest.py               # 文档摄入 CLI
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
│
├── core/                   # 核心基础层
│   ├── document_loader.py  # 多格式文档加载（PDF/MD/CSV）
│   ├── text_splitter.py    # 文本分块
│   ├── embeddings.py       # Google Embedding 封装
│   ├── vectorstore.py      # ChromaDB 向量库管理
│   ├── retriever.py        # 检索器封装
│   ├── multi_query.py      # Multi-Query 多角度检索
│   └── llm.py              # Google Gemini LLM 封装
│
├── agent/                  # Agent 编排层
│   ├── state.py            # LangGraph State 定义
│   ├── nodes.py            # 8 个图节点实现
│   ├── graph.py            # LangGraph 流程图构建
│   └── tools.py            # Web 搜索等外部工具
│
├── prompts/                # Prompt 模板层
│   ├── router.py           # 查询路由
│   ├── grader.py           # 文档相关性评分
│   ├── generator.py        # 带引用的答案生成 + 幻觉检测 + 相关性评估
│   ├── rewriter.py         # 查询重写
│   └── decomposer.py       # 复杂问题拆解
│
├── data/                   # 上传的原始文档
└── vectorstore/            # ChromaDB 持久化存储
```

## 配置说明

编辑 `config.py` 可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini 模型名称 |
| `LLM_TEMPERATURE` | `0.0` | LLM 温度（评分/路由用 0，生成用 0.3） |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding 模型 |
| `CHUNK_SIZE` | `1000` | 文本分块大小 |
| `CHUNK_OVERLAP` | `200` | 分块重叠字符数 |
| `RETRIEVER_TOP_K` | `5` | 每次检索返回的文档数 |
| `MULTI_QUERY_COUNT` | `4` | Multi-Query 子查询数量 |
| `MAX_QUERY_REWRITE_COUNT` | `3` | 最大查询重写次数 |
| `MAX_DECOMPOSITION_STEPS` | `4` | 最大问题拆解子问题数 |

## License

MIT
