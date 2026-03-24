# first_langchain.py
from langchain_ollama import ChatOllama

# 初始化Ollama模型
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.7,
    top_p=0.9,
    num_predict=100
)

# 调用模型
prompt = "请用一句话介绍LangChain是什么。"
response = llm.invoke(prompt)
print("模型回答：", response.content)