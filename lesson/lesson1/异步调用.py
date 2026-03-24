import asyncio
from langchain_ollama import ChatOllama

async def async_generate():
    llm = ChatOllama(model="gpt-oss:20b-cloud", base_url="http://localhost:11434")
    response = await llm.ainvoke("请用一句话介绍异步编程。")
    print(response.content)

# 运行异步函数
asyncio.run(async_generate())