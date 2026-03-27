""""
from langchain.agents import create_react_agent, AgentExecutor
只适用旧版langchain
"""

from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    # 这里本应调用真实天气 API，我们直接返回模拟数据
    weather_data = {
        "北京": "晴，25°C，微风",
        "上海": "多云，28°C，南风3级",
        "广州": "阵雨，30°C，湿度80%",
        "深圳": "雷阵雨，29°C，东南风2级"
    }
    return weather_data.get(city, f"抱歉，暂无 {city} 的天气信息。")

# 1. 初始化模型
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

# 2. 定义工具列表
tools = [get_weather]

# 3. ReAct 提示模板（LangChain 内置了标准模板，我们可以直接使用）
#    但为了灵活性，也可以自定义。这里使用内置模板。
#    注意：内置模板要求工具描述格式为“工具名称: 工具描述, 参数: 参数说明”
#    我们只需传入工具列表即可。
prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)

# 4. 创建代理
agent = create_react_agent(llm, tools, prompt)

# 5. 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印思考过程，方便调试
    handle_parsing_errors=True  # 自动处理解析错误
)

# 6. 运行代理
result = agent_executor.invoke({"input": "北京的天气怎么样？"})
print("最终答案：", result["output"])