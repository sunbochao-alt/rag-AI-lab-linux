# 🧪 AI-Lab-RAG: 基于 DeepSeek 的实验室智能知识库

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)

这是一个为科研团队和开发者打造的轻量级 **RAG (检索增强生成)** 系统。它能将本地 PDF 文档转化为可以交互的智能助手，特别针对轻量级服务器（如 2核 4G）进行了 CPU 运行优化。

---

## ✨ 核心亮点

* **智能文档管理**：支持在线创建分类文件夹，通过侧边栏轻松管理不同研究领域的 PDF。
* **深度适配 DeepSeek**：使用 DeepSeek-V3 核心模型，提供极高性价比的逻辑推理和问答。
* **本地向量检索**：采用 HuggingFace 本地嵌入模型 (`all-MiniLM-L6-v2`)，无需依赖外部嵌入 API。
* **对话上下文感知**：具备历史记忆功能，支持追问和复杂逻辑推导。
* **一键式部署**：专为 Linux 环境优化，内置国内模型镜像加速，解决 HuggingFace 连接难题。

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone [https://github.com/sunbochao-alt/rag-AI-lab-linux.git](https://github.com/sunbochao-alt/rag-AI-lab-linux.git)
cd rag-AI-lab-linux

2. 环境配置 (Conda)
Bash
conda create -n new_lab_rag_env python=3.11
conda activate new_lab_rag_env
pip install -r requirements.txt

3. 设置 API Key
在项目根目录下创建 .env 文件，并填入你的 DeepSeek API Key：

Code snippet
OPENAI_API_KEY=sk-你的DeepSeek密钥

4. 运行系统
Bash
# 使用国内镜像加速下载模型并启动
HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com) streamlit run web_app.py --server.address 0


📂 目录结构说明
web_app.py
系统核心逻辑文件，集成 Streamlit UI、文档切分、向量检索及大模型调用链路。

data/
原始文档存放目录。支持多级文件夹分类，系统会自动识别子目录作为“知识分类”。

db/
自动生成的 ChromaDB 向量数据库目录，用于持久化存储文档向量特征（已在 .gitignore 中忽略）。

requirements.txt
项目依赖清单，包含 LangChain、Streamlit、ChromaDB 等核心库。

📝 开发者备注
内存优化
本项目默认在 CPU 环境运行。代码内置了垃圾回收机制 (gc.collect())，并在重建索引时自动释放内存。

网络补丁
由于国内服务器访问 HuggingFace 限制，项目强制使用 hf-mirror.com 镜像站进行 Embedding 模型下载。

进程管理
如需在后台持久运行，建议使用 nohup：

nohup streamlit run web_app.py --server.address 0.0.0.0 > streamlit.log 2>&1 &