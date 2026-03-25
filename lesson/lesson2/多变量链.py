from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2",
                base_url="http://localhost:11434")

template = "把以下内容翻译成{language}：{text}"
prompt = PromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"language": "英文", "text": "人工智能正在改变世界。"})
print(result)