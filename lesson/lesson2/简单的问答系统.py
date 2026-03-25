from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化模型
llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434",
                temperature=0.2,
                top_p=0.9,
                num_predict=200)

# 定义提示模板
template = """你是一个乐于助人的助手。请用中文回答一下问题，回答简洁明了。
问题：{question}
回答：
"""
prompt = PromptTemplate.from_template(template)
# 构建链
chain = prompt | llm | StrOutputParser()

question = "什么是LangChain？"
answer = chain.invoke({"question": question})
# 运行
print(f"问题：{question}\n回答：{answer}")
