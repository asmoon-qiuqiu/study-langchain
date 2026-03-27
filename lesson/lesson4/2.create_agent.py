"""`create_agent`内置了完整的ReAct逻辑！不用自己写 ReAct 模板"""

# 官方最新版：create_agent + invoke
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    weather_data = {
        "北京": "晴，25°C，微风",
        "上海": "多云，28°C，南风3级",
        "广州": "阵雨，30°C，湿度80%",
        "深圳": "雷阵雨，29°C，东南风2级"
    }
    return weather_data.get(city, "暂无该城市天气")

# 2. 初始化模型
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

# 3. 工具列表
tools = [get_weather]

# 4. 【核心】创建代理（内置ReAct）
# 自动包含你原来的思考/行动/观察逻辑
agent = create_agent(
    model=llm,
    tools=tools,
    # 把你原来的 ReAct 逻辑意图，转为新版系统提示
    system_prompt="""你是智能助手，严格按照 ReAct 流程执行：
    1. Thought：思考如何解决问题
    2. Action：调用工具
    3. Observation：查看工具结果
    4. 最终用自然语言给出 Final Answer
    禁止输出JSON，只给人类可读的中文答案。"""
)

# 5. 运行
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
})

# 6. 输出最终答案
print("最终答案：", result["messages"][-1].content)