from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 模型
llm = OllamaLLM(model="deepseek-v3.1:671b-cloud",
                base_url="http://localhost:11434")

# 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业助手"),
    ("user", "把这句话翻译成英文：{text}")
])

# 解析器
parser = StrOutputParser()

# 组合链
chain = prompt | llm | parser

# 运行
res = chain.invoke({"text": "LangChain 是构建大模型应用最流行的框架"})
print(res)

