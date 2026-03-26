from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory  # 使用 InMemoryChatMessageHistory

# 1. 初始化模型
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# 2. 定义提示模板：使用占位符 {input}
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
    ("human", "{input}")                           # 用户输入占位符，注意是 "{input}"
])

# 3. 创建基本链
chain = prompt | llm | StrOutputParser()

# 4. 定义一个函数，根据 session_id 返回一个消息历史对象
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        # 使用 InMemoryChatMessageHistory，它实现了 BaseChatMessageHistory
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. 使用 RunnableWithMessageHistory 包装链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",          # 输入字典中代表用户消息的键
    history_messages_key="history"       # 提示模板中历史消息的占位符名称
)

# 6. 调用带记忆的链
response = chain_with_history.invoke(
    {"input": "我叫小明"},
    config={"configurable": {"session_id": "user1"}}
)
print(response)

response = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "user1"}}
)
print(response)