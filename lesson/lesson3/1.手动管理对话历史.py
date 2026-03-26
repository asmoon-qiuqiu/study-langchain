from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 初始化模型
llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434")

# 定义提示模板，包含对话历史和当前用户输入
template = """以下是对话历史：
{history}
用户：{input}
AI：
"""
prompt = PromptTemplate.from_template(template)

# 存储对话历史
history = []
def chat(user_input):
    global history
    # 将历史记录格式化为字符串
    history_str = "\n".join(history)
    # 生成提示
    formatted_history = prompt.format(history=history_str, input=user_input)
    # 调用模型
    response = llm.invoke(formatted_history)
    # 更新历史：添加用户消息和AI回复
    history.append(f"用户：{user_input}")
    history.append(f"AI：{history_str}")
    return response

# 模拟对话
print("对话开始（输入exit退出）")
while True:
    user_input = input("你：")
    if user_input == "exit":
        break
    ai_response = chat(user_input)
    print(f"AI:{ai_response}")