from langchain_ollama import OllamaLLM
# 导入最新提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化模型
llm = OllamaLLM(model="deepseek-v3.1:671b-cloud", temperature=0.1)

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业Python编程老师"),
    ("user", "请解释：{topic}")
])

# 3. 组合：链（LCEL：LangChain Expression Language）
chain = prompt | llm

# 4. 运行
result = chain.invoke({"topic": "什么是 LCEL"})
print(result)