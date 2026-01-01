# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
"""
LangChain 1.0 - Context Management (上下文管理)
==============================================
本模块重点讲解：
1. SummarizationMiddleware - 自动摘要中间件（LangChain 1.0 新增）
2. trim_messages - 消息修剪工具
3. 管理对话长度，避免超 token
4. 中间件的使用
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.6
)


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """
    简单的计算器
    :param operation: +-*/
    :param a: 数字
    :param b: 数字
    :return: 字符串
    """
    option = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b,
    }
    result = option.get(operation, 'wrong')(a, b)
    return f"{a} {operation} {b} = {result}"


@tool
def get_used_info(used_id: str) -> str:
    """获取简单的用户消息"""
    users = {
        "123": "张三，25岁，工程师",
        "456": "李四，30岁，设计师"
    }
    return users.get(used_id, '未找到该用户')


# 问题 - 对话历史无限增长
# 解决方案 1 - SummarizationMiddleware（推荐）
def example_1_summarization_middleware():
    """
    示例1：使用 SummarizationMiddleware 自动摘要
    关键：LangChain 1.0 新增的中间件
    当消息数超过阈值时，自动摘要旧消息
    """
    summarizer = SummarizationMiddleware(
        model=model,
        trigger=("tokens", 3000),
    )

    agent = create_agent(
        model=model,
        tools=[calculate],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver(),
        middleware=[summarizer]

    )
    config = {"configurable": {"thread_id": "with_summary"}}

    print("\n进行多轮对话...")
    conversations = [
        "我叫张三，是工程师",
        "我在北京工作",
        "我喜欢编程和阅读",
        "我最近在学习 AI",
        "请总结一下我的信息"
    ]

    for msg in conversations:
        print(f"\n用户: {msg}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config
        )
        print(f"Agent: {response.get('messages')[-1].content}")

    print(f"\n消息数: {len(response['messages'])}")
    print("\n关键点：")
    print("  - SummarizationMiddleware 会自动摘要旧消息")
    print("  - 保持对话历史在可控范围内")
    print("  - 重要信息通过摘要保留")


# 示例 2：手动消息修剪（trim_messages）
def example_2_manual_trimming():
    """
    示例2：使用 trim_messages 手动修剪消息
    适用场景：需要精确控制保留的消息数量
    """
    from langchain_core.messages import trim_messages
    from langchain_core.messages import HumanMessage, AIMessage
    messages = [
        HumanMessage(content="消息 1"),
        AIMessage(content="回复 1"),
        HumanMessage(content="消息 2"),
        AIMessage(content="回复 2"),
        HumanMessage(content="消息 3"),
        AIMessage(content="回复 3"),
        HumanMessage(content="消息 4"),
        AIMessage(content="回复 4"),
    ]
    # 只保留最近 4 条消息
    # 按 token 数裁剪（不严格条数）	max_tokens=N + 合理 token_counter
    # 严格保留最后 N 条消息	max_count=N

    trimmed = trim_messages(
        messages,
        max_tokens=5,  # 或使用 token 数限制
        strategy="last",  # 保留最后的消息
        token_counter=len  # 简单计数器（实际应该用 token 计数）这里其实不会被用到，因为 max_count 优先
    )
    print(f"修剪后消息数: {len(trimmed)}")
    print("\n保留的消息：")
    for msg in trimmed:
        print(f"  {msg.__class__.__name__}: {msg.content}")

    print("\n关键点：")
    print("  - trim_messages 手动控制消息数量")
    print("  - 适合需要精确控制的场景")
    print("  - 需要自己管理修剪逻辑")


def example_3_practical_use():
    """
    简单的实际场景应用
    模拟一个客服机器人
    场景：客服对话可能很长，需要管理上下文
    """
    system_prompt = """
    你是一个客服助手。
    特点：
    - 记住用户说过的话
    - 友好、有耐心
    - 使用 get_user_info 工具查询用户信息时需要用户 ID"""

    summarizer = SummarizationMiddleware(
        model=model,
        trigger=("tokens", 50),
    )
    agent = create_agent(
        model=model,
        tools=[get_used_info,calculate],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        middleware=[]
    )

    config = {'configurable': {'thread_id': '客服1号'}}
    conversations = [
        # "你好，我想咨询一下",
        "我的用户 ID 是 123",
        # "帮我查一下我的信息",
        "我多大来着？",  # 测试记忆
        "帮我算一下 100 * 2 的优惠价。",
        # "谢谢，我的ID是多少？？"
    ]
    for conversation in conversations:
        print(f"用户：{conversation}")
        response = agent.stream({'messages': [{'role': 'human', 'content': conversation}]},
                                config=config)
        for chunk in response:
            if chunk.get('model', {}):
                print(chunk.get('model', {}).get('messages', [])[-1].content)
            elif chunk.get('tools', {}):
                print(chunk.get('tools', {}).get('messages', [])[-1].content)
#trigger=("tokens", 500),  内容如下：
'''
用户：我的用户 ID 是 123
您好！我注意到您提供了用户ID 123。不过，我目前只能使用计算器工具来进行简单的数学运算，或者获取用户信息。如果您需要查询用户123的信息，我可以帮您调用相关工具。

请问您今天需要什么帮助呢？比如：
1. 需要我帮您计算什么数学问题吗？
2. 或者有其他我可以协助您的事情？

我会尽力为您提供友好、耐心的服务！
用户：我多大来着？
要查询您的年龄信息，我需要使用获取用户信息的工具。您刚才提到您的用户ID是123，让我帮您查询一下相关信息。
张三，25岁，工程师
根据查询结果，您的信息是：
- 姓名：张三
- 年龄：25岁
- 职业：工程师

所以您今年25岁！请问还有什么其他需要帮助的吗？
用户：帮我算一下 100 * 2 的优惠价。
我来帮您计算100乘以2的优惠价。
100.0 * 2.0 = 200.0
计算结果：100 × 2 = 200

所以优惠价是200。请问这个价格符合您的预期吗？或者您还需要计算其他折扣或价格吗？'''

# trigger=("tokens", 50),
'''用户：我的用户 ID 是 123
您好！我已经记下了您的用户 ID 是 123。请问有什么可以帮助您的吗？
用户：我多大来着？
我需要使用查询工具来获取您的年龄信息。让我帮您查询一下。
张三，25岁，工程师
根据查询结果，您是张三，25岁，职业是工程师。请问还有其他需要帮助的吗？
用户：帮我算一下 100 * 2 的优惠价。
我来帮您计算一下 100 * 2 的优惠价。
100.0 * 2.0 = 200.0
100 * 2 的优惠价是 200。请问您还需要计算其他优惠价格吗？
'''
"""可以观察到，设计中的 token 阈值在这里扮演了 “回复风格调节器” 的角色。较高的阈值允许系统生成更详尽、更具互动性和引导性的回复，
侧重于提供更好的用户体验和更自然的对话流。而较低的阈值则促使系统生成更精炼、信息密度更高的回复，侧重于直接、高效地完成用户请求。
这体现了在资源（如上下文长度、生成时间）和回复质量（如友好度、完整性）之间进行权衡的一种设计思路。
"""
def main():
    try:
        example_1_summarization_middleware()
        example_2_manual_trimming()
        example_3_practical_use()
    except Exception as e:
        print(e)
        print("""
         策略对比：

         1. 不做处理（默认）
            优点：保留完整历史
            缺点：会超 token、成本高
            适用：短对话

         2. SummarizationMiddleware（推荐）
            优点：
            - 自动化，无需手动管理
            - 保留重要信息（通过摘要）
            - 平滑过渡
            缺点：
            - 摘要可能丢失细节
            - 额外的摘要成本
            适用：长对话、需要保留上下文

         3. trim_messages（手动修剪）
            优点：
            - 精确控制
            - 简单直接
            - 无额外成本
            缺点：
            - 旧消息完全丢失
            - 可能断开上下文
            适用：只需要最近 N 轮

         4. 滑动窗口（自定义）
            优点：
            - 保留系统消息 + 最近消息
            - 可控成本
            缺点：
            - 需要自己实现
            适用：有明确规则的场景

         推荐方案：
         - 短对话（<10轮）：不处理
         - 中长对话：SummarizationMiddleware
         - 只要最近几轮：trim_messages
             """)


if __name__ == '__main__':
    main()

