from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import  ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import  Chroma
from langchain_core.runnables import RunnablePassthrough

# 1. 创建测试文档
with open("知识.txt", "w", encoding="utf-8") as f:
    f.write("""
    LangChain 1.2.13 是 2026 年最新稳定版。
    它支持 LCEL 语法，支持 Ollama 本地模型。
    LangGraph 是它的官方工作流框架。
""")

# 2. 加载 + 切分文档
loader = TextLoader("知识.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
splits = splitter.split_documents(docs)

# 3. 向量库（Chroma）
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b",
                              base_url="http://localhost:11434")  # 本地嵌入
db = Chroma.from_documents(splits, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 4. RAG 提示词
template = """
根据下面的资料回答问题：
资料：{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. RAG 链
llm = OllamaLLM(model="qwen3.5:4b",
                base_url="http://localhost:11434",
                temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 测试
question = "LangChain 1.2.13 支持什么？"
print(rag_chain.invoke(question))
