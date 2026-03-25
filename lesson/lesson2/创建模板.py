from langchain_core.prompts import PromptTemplate

# 定义模板，用 {variable} 表示变量
template = "请用一句话介绍{subject}是什么。"
prompt_template = PromptTemplate.from_template(template)

# 填充变量
formatted_prompt = prompt_template.format(subject="LangChain")
print(formatted_prompt)
# 输出：请用一句话介绍LangChain是什么。