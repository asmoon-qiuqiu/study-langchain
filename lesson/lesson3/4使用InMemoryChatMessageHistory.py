from langchain_core.chat_history import InMemoryChatMessageHistory

# 直接创建历史
history = InMemoryChatMessageHistory()

# 添加消息
history.add_user_message("我叫小明")
history.add_ai_message("你好小明！很高兴认识你。")
history.add_user_message("我叫什么名字？")

# 查看所有消息
print(history.messages)  # 消息列表