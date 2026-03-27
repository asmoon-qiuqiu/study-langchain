from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 工具1：天气
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    weather_map = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "阵雨，30°C"
    }
    return weather_map.get(city, "暂无该城市天气信息")

# 工具2：计算器
@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2+3*4'"""
    try:
        return str(eval(expression))
    except:
        return "表达式无效"

# 模型 + 代理
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)
tools = [get_weather, calculator]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是智能助手，需要计算就调用calculator，需要天气就调用get_weather，只用自然语言回答。"
)
# 交互菜单：输入数字选择
while True:
    print("\n===== 智能代理菜单 =====")
    print("1 → 数学计算")
    print("2 → 查询天气")
    print("0 → 退出")

    choice = input("请输入数字：")

    if choice == "1":
        expr = input("请输入数学表达式（例如 3+5*2）：")
        res = agent.invoke({
            "messages": [HumanMessage(content=f"计算：{expr}")]
        })
        print("✅ 结果：", res["messages"][-1].content)

    elif choice == "2":
        city = input("请输入城市名称（北京/上海/广州）：")
        res = agent.invoke({
            "messages": [HumanMessage(content=f"{city}的天气如何？")]
        })
        print("✅ 天气：", res["messages"][-1].content)

    elif choice == "0":
        print("退出成功！")
        break

    else:
        print("输入无效，请重新输入！")