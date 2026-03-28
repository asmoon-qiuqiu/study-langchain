from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 实例文档
with open("company.txt", "w", encoding="utf-8") as f:
    f.write("""
    蓝桥科技成立于2020年，总部位于杭州，是一家专注于人工智能解决方案的创新型公司。
    公司核心产品包括：
    1. 智能客服系统：基于大语言模型，帮助企业自动化处理客户咨询。典型应用场景公司客服服务系统。
    2. 数据分析平台：提供实时业务洞察与预测分析。
    3. 文档智能处理：支持从海量文档中提取关键信息。
    公司目前拥有200名员工，其中研发人员占比70%。客户覆盖金融、医疗、教育等行业。
    """)

# 加载文档
loader = TextLoader("company.txt", encoding="utf-8")
documents  = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 每个片段的最大字符数
    chunk_overlap=50,   # 相邻片段的重复字符数
    # separators=[]     # 默认：["\n\n", "\n", " ", ""]-先按段落切，再按句子切，最后按单词切
)
chunks = text_splitter.split_documents(documents)
print(f"分割为{len(chunks)}")

# 初始化嵌入模型（Ollama 支持 embedding 模型）
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    base_url="http://localhost:11434"
)

# 创建向量数据库（自动嵌入并存储）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化存储"
)
print("向量数据库创建完成")

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 检索最相关的3个片段

# 定义格式化函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 初始化大模型
llm = ChatOllama(
    model="llama3.2",  # 生成模型
    base_url="http://localhost:11434",
    temperature=0
)

# 定义带记忆的提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。请根据检索内容回答问题，不知道就说不知道。"),
    MessagesPlaceholder(variable_name="history"), # 历史消息将插入此处
    ("system", "检索到的相关文档：\n{context}"),
    ("human", "{question}"),
])

# 构建 RAG 链
# 定义一个可以接收字典并返回 context 的组件
retrieve_and_format = RunnablePassthrough.assign(
    context=(RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))))
) #RunnableWithMessageHistory会把输入包装成字典，retriever传的是字符串所以需要question提取出来
rag_chain = (
    retrieve_and_format
    | prompt
    | llm
    | StrOutputParser()
)

# 内存存储对话历史（多个用户会话）
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 包装链，使它能自动处理历史消息
rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 测试：多轮对话（能记住上下文）

session_id = "user123"

# 第一轮
response1 = rag_with_memory.invoke(
    {"question": "蓝桥科技的核心产品有哪些？"},
    config={"configurable": {"session_id": session_id}}
)
print("用户: 蓝桥科技的核心产品有哪些？")
print("AI:", response1)

# 第二轮（追问）
response2 = rag_with_memory.invoke(
    {"question": "其中哪一款产品是基于大语言模型的？"},
    config={"configurable": {"session_id": session_id}}
)
print("\n用户: 其中哪一款产品是基于大语言模型的？")
print("AI:", response2)

# 第三轮（不直接依赖文档，但依赖历史）
response3 = rag_with_memory.invoke(
    {"question": "那款产品的典型应用场景有哪些？"},
    config={"configurable": {"session_id": session_id}}
)
print("\n用户: 那款产品的典型应用场景有哪些？")
print("AI:", response3)