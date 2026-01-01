# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
"""
1. 清晰的 docstring
   @tool
   def search_products(query: str, max_results: int = 10) -> str:
       '''
       在产品数据库中搜索产品

       参数:
           query: 搜索关键词
           max_results: 最大返回数量，默认10

       返回:
           产品列表的JSON字符串
       '''

2. 明确的参数类型
   - 使用类型注解：str, int, float, bool
   - 可选参数用 Optional[类型]

3. 返回字符串
   - 工具应该返回 str（AI 最容易理解）
   - 复杂数据可以返回 JSON 字符串

4. 错误处理
   - 在工具内部捕获异常
   - 返回友好的错误消息

5. 功能单一
   - 一个工具做一件事
   - 不要把多个功能塞进一个工具
    """

from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tools.web_search import web_search
# from langchain_core.tools import tool_calls

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.5,
)

"""
示例 将工具绑定到模型

这是让 AI 使用工具的第一步
"""
print("\n" + "="*70)
print("示例：工具绑定到模型（预览）")
print("="*70)

# 绑定工具到模型
model_with_tools = model.bind_tools([web_search])

print("模型已绑定工具：")
print("web_search")


# 调用模型（模型可以选择使用工具）
print("\n测试：AI 是否会调用网站搜索工具？")
response = model_with_tools.invoke("python")

# 检查模型是否要求调用工具
if response.tool_calls:
    print(f"\n✅ AI 决定使用工具！")
    print(f"工具调用: {response.tool_calls}")
else:
    print(f"\nℹ️ AI 直接回答（未使用工具）")
    print(f"回复: {response.content}")