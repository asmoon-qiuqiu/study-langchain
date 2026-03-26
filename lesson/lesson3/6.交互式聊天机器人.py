from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 初始化模型（ChatOllama）
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.7,
    num_predict=200
)

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。请用中文回答。"),
    MessagesPlaceholder(variable_name="history"), # 历史消息占位符
    ("human", "{input}") # 用户输入占位符
])

# 创建基本链
chain = prompt | llm | StrOutputParser()

# 存储不同会话的历史
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 包装链以支持记忆
chain_with_history =RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 启动交互
print("聊天机器人已启动（输入exit退出）")
session_id = "default_session"   # 固定会话ID，可根据需要修改

while True:
    user_input = input("你：")
    if user_input.lower() == "exit":
        break
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"AI: {response}")