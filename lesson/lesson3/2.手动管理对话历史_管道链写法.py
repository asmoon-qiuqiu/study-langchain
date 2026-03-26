from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 初始化模型
llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434")

# 定义提示模板
template = """以下是对话历史：
{history}
用户：{input}
AI：
"""
prompt = PromptTemplate.from_template(template)

# 创建管道链
chain = prompt | llm

# 存储对话历史
history = []

def chat(user_input):
    global history
    # 格式化历史
    history_str = "\n".join(history)

    # 管道写法必须传字典！不能传字符串
    response = chain.invoke({
        "history": history_str,
        "input": user_input
    })
    history.append(f"用户：{user_input}")
    history.append(f"AI：{response}")  # 这里必须是 response！

    return response


# 模拟对话
print("对话开始（输入exit退出）")
while True:
    user_input = input("你：")
    if user_input == "exit":
        break
    ai_response = chat(user_input)
    print(f"AI:{ai_response}")