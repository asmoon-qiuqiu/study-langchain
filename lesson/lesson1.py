# # 导入聊天模型
# from langchain_deepseek import  ChatDeepSeek
# from dotenv import load_dotenv
#
# load_dotenv()
# # 初始化模型
# llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.7)
#
# # 调用模型（输入是消息列表，包含角色和内容）
# messages = [
#     ("system", "你是一个友好的编程老师，回答简洁易懂"),
#     ("user", "什么是 LangChain？")
# ]
# response = llm.invoke(messages)
#
# # 输出结果
# print("模型回答：", response.content)


# 从 langchain-ollama 导入最新 Ollama 封装
from langchain_ollama import OllamaLLM

# 1. 初始化模型（适配 LangChain 1.2.13）
llm = OllamaLLM(
    model="gpt-oss:20b-cloud",  # 你本地 ollama 里的模型名
    base_url="http://localhost:11434",  # Ollama 默认地址
    temperature=0.1,  # 0=更严谨 1=更创意
)

# 2. 最简单调用
response = llm.invoke("你好，请用一句话介绍 LangChain")

# 3. 输出结果
print(response)