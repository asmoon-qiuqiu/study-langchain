from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    weather_map = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "阵雨，30°C"
    }
    return weather_map.get(city, "暂无该城市天气信息")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2+3*4'"""
    try:
        return str(eval(expression))
    except:
        return "表达式无效"

# 模型
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)
tools = [get_weather, calculator]

# 调用create_agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是智能助手，需要计算就调用calculator，需要天气就调用get_weather，只用自然语言回答。"
)

# 调用格式
# 必须传 {"messages": [...]}
print("--- 计算测试 ---")
res1 = agent.invoke({
    "messages": [HumanMessage(content="帮我算一下 (3+5)*2 等于多少？")]
})
print("答案：", res1["messages"][-1].content)

print("\n--- 天气测试 ---")
res2 = agent.invoke({
    "messages": [HumanMessage(content="上海的天气如何？")]
})
print("答案：", res2["messages"][-1].content)