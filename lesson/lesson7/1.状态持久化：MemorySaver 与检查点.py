from typing import Annotated, List, TypedDict

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
import operator
import re
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,END


# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 2. 初始化模型、向量库、组件
llm = ChatOllama(model="qwen3.5:4b",
                 base_url="http://localhost:11434", temperature=0)

# RAG 组件
loader = TextLoader("company.txt", encoding="utf-8")
document = loader.load()
# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
).split_documents(document)

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://localhost:11434")
vectorstore = Chroma(embedding_function=embeddings,
                     persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 定义工具
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

@tool
def retrieve_docs(query: str) -> str:
    """从知识库中检索相关文档"""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [get_weather, retrieve_docs]
tool_map = {tool.name: tool for tool in tools}

# 4. 解析函数
def parse_action(text: str):
    """只匹配 Action 和 Action Input 行，忽略其他内容"""
    action_match = re.search(r"Action:\s*([^\n]+)", text)
    input_match = re.search(r"Action Input:\s*([^\n]+)", text)
    if action_match and input_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        return action, action_input
    return None, None

# 5. 路由函数
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

# 6. 节点函数
def agent_node(state: AgentState):
    """调用模型，输出可能包含动作指令"""
    system_prompt = """
    你是一个智能助手，可以使用以下工具：
    - get_weather: 获取天气，输入城市名（中文）
    - retrieve_docs: 从知识库检索，输入问题

    **重要格式规则：**

    1. 当你需要调用工具时，必须输出：
       Action: 工具名
       Action Input: 输入内容

    2. 当你从系统接收到 Observation 后，必须输出最终答案，格式为：
       Final Answer: 你的回答

    3. 如果你不需要调用任何工具，可以直接输出 Final Answer。

    注意：只输出以上格式，不要添加任何解释或额外文字。一旦输出 Final Answer，流程就会结束。
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

def tools_node(state: AgentState):
    last_message = state["messages"][-1]
    content = last_message.content
    action, action_input = parse_action(content)
    if action and action in tool_map:
        result = tool_map[action].invoke(action_input)
    else:
        result = f"未知工具或未提供正确格式: {content}"
    return {"messages": [HumanMessage(content=f"Observation: {result}")]}

# 7. 构建流程图
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
workflow.add_edge("tools", "agent")
# 编译时指定检查点
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 启动一个会话
config = {"configurable": {"thread_id": "session_1"}}

# 第一轮
print("="*10, "第一轮对话", "="*10)
res1 = app.invoke({"messages": [HumanMessage(content="核心产品有哪些？")]}, config)
for msg in res1["messages"]:
    print(f"{msg.type}: {msg.content}\n")

# 第二轮（带记忆）
print("\n" + "="*10, "第二轮对话", "="*10)
res2 = app.invoke({"messages": [HumanMessage(content="哪一款产品基于什么模型？")]}, config)
for msg in res2["messages"]:
    print(f"{msg.type}: {msg.content}\n")

# 第三轮（带记忆）
print("\n" + "=" * 10, "第三轮对话", "=" * 10)
res3 = app.invoke({"messages": [HumanMessage(content="我刚才问了什么？")]}, config)
print(res3["messages"][-1].content)