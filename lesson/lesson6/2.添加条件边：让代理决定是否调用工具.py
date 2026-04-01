from typing import TypedDict, List, Annotated
import operator
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain.tools import tool

# ==========================
# 1. 定义状态（Agent 的记忆）
# ==========================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ==========================
# 2. 初始化大模型
# ==========================
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

# ==========================
# 3. 系统提示词（给 AI 的规则）
# ==========================
SYSTEM_PROMPT = """你是一个智能助手，可以使用 get_weather 工具查询天气。
当用户询问天气时，你必须严格按照以下格式输出：

Action: get_weather
Action Input: 城市名

然后系统会返回天气信息，你看到 Observation 后，必须输出最终答案，格式为：
Final Answer: 完整的回答

如果你不需要使用工具，可以直接输出 Final Answer。
不要输出任何其他内容。
"""

# ==========================
# 4. 工具定义
# ==========================
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    weather_data = {
        "北京": "晴，25°C，微风",
        "上海": "多云，28°C，南风3级",
        "广州": "阵雨，30°C，湿度80%",
        "深圳": "雷阵雨，29°C，东南风2级"
    }
    return weather_data.get(city, f"抱歉，暂无 {city} 的天气信息。")

# ==========================
# 5. 工具解析函数
# ==========================
def parse_action(text: str):
    """只匹配 Action 和 Action Input 行，忽略其他内容"""
    action_match = re.search(r"Action:\s*([^\n]+)", text)
    input_match = re.search(r"Action Input:\s*([^\n]+)", text)
    if action_match and input_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        return action, action_input
    return None, None

# ==========================
# 6. 路由函数（判断下一步怎么走）
# ==========================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    content = last_message.content
    # 如果已经包含 Final Answer，直接结束
    if "Final Answer:" in content:
        return "end"
    action, _ = parse_action(content)
    if action:
        return "tools"
    else:
        return "end"

# ==========================
# 7. AI 节点
# ==========================
def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

# ==========================
# 8. 工具执行节点
# ==========================
def tools_node(state: AgentState):
    """执行工具并返回结果作为 Observation"""
    last_message = state["messages"][-1]
    content = last_message.content
    action, action_input = parse_action(content)
    if action == "get_weather":
        result = get_weather.invoke(action_input)
    else:
        result = f"未知工具: {action}"
    # 将观察结果作为新消息追加
    return {"messages": [HumanMessage(content=f"Observation: {result}")]}

# ==========================
# 9. 构建流程图
# ==========================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)

workflow.add_edge("tools", "agent")
app = workflow.compile()

# ==========================
# 10. 运行代码
# ==========================
initial_state = {
    "messages": [HumanMessage(content="北京的天气怎么样？")]
}
final_state = app.invoke(initial_state)

for msg in final_state["messages"]:
    print(f"{msg.type}: {msg.content}")