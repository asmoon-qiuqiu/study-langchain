import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


async def async_chain():
    llm = OllamaLLM(model="llama3.2",
                    base_url="http://localhost:11434")
    prompt = PromptTemplate.from_template("用一句话解释{concept}。")
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"concept": "异步编程"})
    print(result)

asyncio.run(async_chain())

