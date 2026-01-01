# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")
base_url = getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    max_tokens=200,
    temperature=1,
    timeout=15.0,
)


# 其他简单练习
def exercise_1_chat_prompt_template():
    # 格式1：纯字符串
    res_one = model.invoke('你好')
    print(res_one)

    # 格式2：字典列表（推荐）
    question = '深度学习是什么'
    messages = [
        {'role': 'system', 'content': '你是一个简洁的助手，回答限制在30字以内'},
        {'role': 'user', 'content': question},
    ]
    res_two = model.invoke(messages)
    print(res_two)

    # 格式3：消息对象
    message = [
        SystemMessage(content='你是一个简洁的助手，回答限制在30字以内'),
        HumanMessage(content=question)
    ]
    res_three = model.invoke(message)
    print(res_three)


# 实例：构建一个简单的聊天机器人
def chat_prompt_template():
    chat_template = [
        {'role': 'system', 'content': '你是一个友好、幽默的助手，喜欢帮助用户'}
    ]

    total_tokens_used = 0
    turn = 0

    list_questions = [
        "你好！",
        "你能做什么？简短回答",
        "告诉我一个编程笑话",
    ]
    return chat_template, total_tokens_used, turn, list_questions


def chat_invoke():
    turn = chat_prompt_template()[2]
    chat_template_list = chat_prompt_template()[0]
    list_questions = chat_prompt_template()[3]
    for question in list_questions:
        chat_template_list.append(
            {'role': 'user', 'content': question}
        )

        response = model.invoke(chat_template_list)
        print()
        print(response.content)
        print()
        turn += 1
        print(turn, tokens_used(response))
        chat_template_list.append({'role': 'assistant', 'content': response.content})


def tokens_used(response):
    total_tokens_used = chat_prompt_template()[1]
    usage = response.response_metadata['token_usage']
    total_tokens_used = total_tokens_used + usage['total_tokens']
    return total_tokens_used


def main():
    try:
        chat_invoke()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # exercise_1_chat_prompt_template()
    main()
