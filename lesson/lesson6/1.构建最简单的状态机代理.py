from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaLLM,ChatOllama
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """代理的状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 对话历史，支持追加
    # 我们还可以添加其他字段，如中间结果、检索到的文档等

llm = ChatOllama(model="llama3.2",
                base_url="http://localhost:11434",
                temperature=0)

def agent_node(state: AgentState):
    """调用模型，将模型的输出追加到消息列表"""
    # 提取最近的消息作为上下文（这里简单起见，传入所有消息）
    messages = state["messages"]
    # 调用模型
    response = llm.invoke(messages)
    # 返回更新后的状态
    return {"messages": [AIMessage(content=response.content)]}

def tools_node(state: AgentState):
    """执行工具（占位，后续实现）"""
    # 暂时什么也不做，返回原状态
    return {}


# 创建图，指定状态类型
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加边：agent -> END（简单线性流程）
workflow.add_edge("agent", END)

# 编译图
app = workflow.compile()

initial_state = {
    "messages": [HumanMessage(content="你好，请介绍一下你自己。")]
}

# 执行图
final_state = app.invoke(initial_state)

# 输出最后一条 AI 消息
print(final_state["messages"])
print(final_state["messages"][-1])
print(final_state["messages"][-1].content)