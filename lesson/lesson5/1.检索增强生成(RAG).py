from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 实例文档
with open("company.txt", "w", encoding="utf-8") as f:
    f.write("""
    蓝桥科技成立于2020年，总部位于杭州，是一家专注于人工智能解决方案的创新型公司。
    公司核心产品包括：
    1. 智能客服系统：基于大语言模型，帮助企业自动化处理客户咨询。
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

# 初始化大模型
llm = ChatOllama(
    model="llama3.2",  # 生成模型
    base_url="http://localhost:11434",
    temperature=0
)

# 定义提示模板
template = """你是一个智能助手。请根据以下检索到的内容回答用户问题。
如果不知道答案，就说不知道。
检索内容：
{context}
用户问题：{question}
回答："""

prompt = ChatPromptTemplate.from_template(template)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 检索最相关的3个片段

# 构建 RAG 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 提问测试
question = "蓝桥科技的核心产品有哪些？"
response = rag_chain.invoke(question)
print("问题:", question)
print("回答:", response)

# 另一个问题
question2 = "公司总部在哪里？"
response2 = rag_chain.invoke(question2)
print("\n问题:", question2)
print("回答:", response2)

# 超出文档范围的问题
question3 = "公司的年营收是多少？"
response3 = rag_chain.invoke(question3)
print("\n问题:", question3)
print("回答:", response3)