from langchain_classic.memory import ConversationBufferMemory


memory = ConversationBufferMemory()

# 添加对话
memory.chat_memory.add_user_message("我叫小明")
memory.chat_memory.add_ai_message("你好小明！很高兴认识你。")
memory.chat_memory.add_user_message("我叫什么名字？")

# 获取历史（返回字符串）
history_str = memory.load_memory_variables({})["history"]
print(history_str)

"""已被InMemoryChatMessageHistory替代"""