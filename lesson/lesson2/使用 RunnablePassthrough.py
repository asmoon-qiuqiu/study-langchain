from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434")

# 模板中包含一个变量 question
template = "请回答以下问题：{question}"
prompt = PromptTemplate.from_template(template)

# 定义一个链，先传递原始输入（原样保留），然后生成提示，调用模型，最后解析
chain = (
    {"question": RunnablePassthrough()}  # 这一步把原始输入作为字典传递
    | prompt
    | llm
    | StrOutputParser()
)

# 但上面的写法其实可以简化，因为 prompt 本身就能直接接受字典。更常见的模式是直接：
chain = prompt | llm | StrOutputParser()
# 无需额外包装，因为 prompt 会自动提取字典中需要的字段。

result = chain.invoke({"question": "什么是langchain。"})
print(result)