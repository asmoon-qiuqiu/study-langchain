from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1.初始化模型
llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434")

# 2.创建模板
template = "请用一句话介绍{subject}是什么。"
prompt = PromptTemplate.from_template(template)

# 3.创建链：将模板、模型和输出解析器串联
chain = prompt | llm | StrOutputParser()

# 4.执行链，传入变量
result = chain.invoke({"subject": "LangChain"})
print(result)