import sys
import os
import shutil
import gc
from operator import itemgetter
import time  
# -----------------------------------------------------------------------------
# 0. 配置国内镜像 & Linux 兼容补丁 (最优先执行)
# -----------------------------------------------------------------------------
# 强制使用 Hugging Face 国内镜像 (解决 Network unreachable 问题)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Linux ChromaDB 补丁
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
import streamlit as st
from dotenv import load_dotenv

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_openai import ChatOpenAI

# Vector Database
from langchain_community.vectorstores import Chroma

# Document Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Core Primitives (LCEL)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

# -----------------------------------------------------------------------------
# 2. 初始化配置
# -----------------------------------------------------------------------------
load_dotenv() # 加载 .env 文件

DATA_PATH = "./data"
DB_PATH = "./db"

os.makedirs(DATA_PATH, exist_ok=True)
st.set_page_config(page_title="本地 RAG 知识库 (DeepSeek Core版)", layout="wide")

# -----------------------------------------------------------------------------
# 3. 核心逻辑：向量数据库构建 (已修复文件锁 Bug)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def rebuild_vector_db():
    # 1. 强制清理旧资源
    # 尝试访问并清理可能存在的旧变量，强制断开数据库连接
    if 'vectorstore' in globals():
        del globals()['vectorstore']
    
    # 强制垃圾回收，释放内存中的文件句柄
    gc.collect()
    
    # 🛑 关键修复：暂停 1 秒，等待操作系统完全释放 SQLite 文件锁
    # Linux 上的文件删除有时是异步的，不等待会导致 "ReadOnly" 或 "Locked" 错误
    time.sleep(1)

    # 2. 清理旧数据库文件夹
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            # 再次等待文件系统同步
            time.sleep(0.5)
        except Exception as e:
            st.error(f"清理旧数据库失败 (文件可能仍被占用，请重启服务): {e}")
            return None

    # 3. 扫描数据目录
    documents = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    categories = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    
    if not categories:
        return None

    status_text = st.empty()
    status_text.info("正在扫描文档并重建知识库...")

    for category in categories:
        cat_path = os.path.join(DATA_PATH, category)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(".pdf")]
        
        for file in files:
            file_path = os.path.join(cat_path, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["source"] = file
                documents.extend(docs)
            except Exception as e:
                st.warning(f"无法加载文件 {file}: {e}")

    if not documents:
        status_text.warning("未找到任何 PDF 文档。")
        return None

    # 4. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 5. 重新初始化 Embeddings 和 数据库
    # 使用 Hugging Face 国内镜像下载模型 (如果还没下载过)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
            #persist_directory=DB_PATH
        )
    except Exception as e:
        st.error(f"创建向量数据库失败: {e}")
        # 如果报错，尝试再清理一次以便下次重试
        shutil.rmtree(DB_PATH, ignore_errors=True)
        return None
    
    status_text.success(f"重建完成！共 {len(splits)} 个切片。")
    return vectorstore

vectorstore = rebuild_vector_db()

# -----------------------------------------------------------------------------
# 4. 侧边栏：数据管理
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂️ 知识库管理")
    current_categories = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    
    new_cat = st.text_input("新建分类文件夹", placeholder="例如：Lab_Protocols")
    if st.button("创建分类"):
        if new_cat:
            target_dir = os.path.join(DATA_PATH, new_cat)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                st.success(f"已创建: {new_cat}")
                st.rerun()

    st.markdown("---")
    selected_cat_upload = st.selectbox("选择上传分类", ["(请选择)"] + current_categories)
    uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    
    if st.button("💾 保存并更新"):
        if selected_cat_upload != "(请选择)" and uploaded_files:
            save_dir = os.path.join(DATA_PATH, selected_cat_upload)
            for uploaded_file in uploaded_files:
                with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.cache_resource.clear()
            st.rerun()

    if st.button("🔄 强制刷新"):
        st.cache_resource.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 5. 主界面：LCEL RAG 逻辑
# -----------------------------------------------------------------------------
st.title("🧪 实验室助手AI (DeepSeek版 BY 孙博超)")

search_category = st.selectbox("🔍 搜索范围", ["全部"] + current_categories, index=0)

if not vectorstore:
    st.info("👈 请先上传文档初始化知识库。")
    st.stop()

# --- LLM ---
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.1
)

# --- Retriever ---
search_kwargs = {"k": 4}
if search_category != "全部":
    search_kwargs["filter"] = {"category": search_category}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

# --- 辅助函数 ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Step 1: 历史回溯 (History Aware) ---
contextualize_q_system_prompt = (
    "给定聊天历史和用户问题，请将问题重写为一个独立的问题，"
    "使其无需上下文即可理解。直接输出重写后的问题，不要解释。"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_chain = contextualize_q_prompt | llm | StrOutputParser()

# --- 修正后的 RunnableBranch ---
# 逻辑：如果有 chat_history，走 history_aware_chain；否则直接透传 input
retrieval_chain = RunnableBranch(
    (lambda x: len(x.get("chat_history", [])) > 0, history_aware_chain | retriever),
    itemgetter("input") | retriever # <--- 修正点：默认分支直接写 Runnable，不要加元组
)

# --- Step 2: 问答生成 (Stuff Documents) ---
qa_system_prompt = (
    "你是一位严谨的实验室助手。请仅根据提供的上下文(Context)回答用户的问题。"
    "如果上下文中没有答案，请直接说明不知道。\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 完整的 RAG 链
rag_chain = (
    {
        "context": retrieval_chain,
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")
    }
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 用于引用显示的单独链
source_retrieval_chain = retrieval_chain

# -----------------------------------------------------------------------------
# 6. UI 交互
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 构建历史
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        try:
            # 1. 生成回答
            full_response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            message_placeholder.markdown(full_response)

            # 2. 显示引用
            retrieved_docs = source_retrieval_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })

            if retrieved_docs:
                with st.expander("📚 参考来源"):
                    seen = set()
                    for doc in retrieved_docs:
                        sid = f"{doc.metadata.get('category')} - {doc.metadata.get('source')}"
                        if sid not in seen:
                            st.markdown(f"- `{sid}`")
                            seen.add(sid)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"发生错误: {e}")