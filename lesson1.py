# 注意：导入的是 ChatOllama 而不是 Ollama
from langchain_community.chat_models import ChatOllama

# 1. 创建 Chat 模式的 Ollama 模型
llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud",
    temperature=0.7  # 可以正常设置温度了
)

# 2. 调用模型（返回 AIMessage 对象，支持 .content）
response = llm.invoke("你好，你是谁？")

# 3. 输出结果（和 OpenAI 写法完全一致）
print(response.content)