from langchain_core.prompts import  PromptTemplate

template_multi = "请把以下内容翻译成{language}：\n{text}"
multi_template = PromptTemplate.from_template(template_multi)

formatted = multi_template.format(language="英文", text="你好，世界")
print(formatted)
# 输出：请把以下内容翻译成英文：
# 你好，世界