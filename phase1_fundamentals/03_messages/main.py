# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
# 消息类型与对话管理
from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.5,
)


# 三种消息类型：SystemMessage, HumanMessage, AIMessage
def example_1_message_type():
    # 重点：字典格式vs消息对象（推荐用字典）
    # 1消息对象（啰嗦）
    message1 = [
        SystemMessage(content='你是一个AI导师。'),
        HumanMessage(content='概括一下：什么是机器学习。')
    ]

    response1 = model.invoke(message1)
    print(response1.content)

    # 2.字典格式
    message2 = [
        {'role': 'system', 'content': '你是一个AI导师。'},
        {'role': 'human', 'content': '概括一下：什么是深度学习。'},
    ]
    response2 = model.invoke(message2)
    print(response2.content)


def example_2_conversation_history():
    # 系统提示 + 第一轮对话
    conversation = [{'role': 'system', 'content': '你是一位AI大模型工程师。'},
                    {'role': 'human', 'content': '概括性地解释一下深度学习。'}]

    response1 = model.invoke(conversation)
    print(response1.content)
    # 加入对话历史
    conversation.append({'role': 'assistant', 'content': response1.content})

    # 第二轮对话
    conversation.append({'role': 'human', 'content': 'python数据分析的核心是什么？'})
    response2 = model.invoke(conversation)
    print(response2.content)
    # 加入对话历史
    conversation.append({'role': 'assistant', 'content': response2.content})

    # 第三轮对话
    conversation.append({'role': 'human', 'content': '我的第一个问题是什么？'})
    response3 = model.invoke(conversation)
    print(response3.content)
    # 加入对话历史
    conversation.append({'role': 'assistant', 'content': response3.content})
    print(conversation)


def example_3_optimise_history():
    # 只保留最近N条对话历史
    def keep_recent_history(messages, max_pairs=2):
        # ststem与对话分离
        system_mes = [message for message in messages if message.get('role') == 'system']
        conversation_mes = [message for message in messages if message.get('role') != 'system']

        # 保留最近2条
        max_message_count = max_pairs * 2
        optimised_mes = conversation_mes[-max_message_count:]
        return system_mes + optimised_mes

    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "第1个问题"},
        {"role": "assistant", "content": "第1个回答"},
        {"role": "user", "content": "第2个问题"},
        {"role": "assistant", "content": "第2个回答"},
        {"role": "user", "content": "第3个问题"},
        {"role": "assistant", "content": "第3个回答"},
        {"role": "user", "content": "第4个问题"},
        {"role": "assistant", "content": "第4个回答"},
        {"role": "user", "content": "第5个问题"},
    ]

    # 优化技巧：当对话太长时，只保留最近的几轮即可
    optimised_mes = keep_recent_history(messages, max_pairs=2)
    response = model.invoke(optimised_mes)
    print(response.content)


def example_4_practice_save_history():
    conversation = [
        {"role": "system", "content": "你是一个友好的助手"}
    ]

    questions = [
        "我叫李明，今年25岁",
        "我喜欢编程",
        "我叫什么名字？",
        "我今年多大？",
        "我喜欢什么？"
    ]

    for i, q in enumerate(questions, 1):
        ''' 
        enumerate()的用法
        for 索引, 值 in enumerate(可迭代对象):
        enumerate(iterable, start=0)
        iterable: 可迭代对象（列表、元组、字符串等）
        start: 索引的起始值，默认为 0
        '''
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {q}")

        conversation.append({"role": "user", "content": q})
        response = model.invoke(conversation)

        print(f"AI: {response.content}")
        conversation.append({"role": "assistant", "content": response.content})


def main():
    try:
        # example_1_message_type()
        # example_2_conversation_history()
        # example_3_optimise_history()
        example_4_practice_save_history()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
