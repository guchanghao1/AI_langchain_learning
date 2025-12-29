# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 实例：构建一个简单的聊天机器人
load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model="deepseek-chat",
    api_key=api_key,
    max_tokens=200,
    temperature=1,
    timeout=15.0,
)


def create_syatem_template():
    # 创建系统提示
    system_template = "你是一个{feature}的{profession}。请保持{style}的回复风格。"
    # 创建完整的提示词模板
    chat_template = ChatPromptTemplate([
        ('system', system_template),
        MessagesPlaceholder(variable_name='chat_history'),
        # 此提示词模板负责在特定位置添加消息列表。 在上面的ChatPromptTemplate中，我们看到如何格式化两个消息，每个消息都是一个字符串。
        # 但是如果我们希望用户传入一个消息列表，并将其插入到特定位置呢？ 这就是如何使用MessagesPlaceholder。
        ('user', '{input}')
    ])
    return chat_template


def chat_prompt_template():
    chat_template = create_syatem_template()

    chat_history = []
    # 格式化系统参数
    template_params = {
        'feature': '幽默、友好',
        'profession': 'AI工程师',
        'style': '轻松愉快',
    }

    list_questions = [
        "你好！",
        "你能做什么？简短回答",
        "解释一下langchain中的LCEL",
    ]
    return chat_template, chat_history, template_params, list_questions


def chat_invoke():
    turn = 0
    total_tokens = 0
    chat_template, chat_history, template_params, list_questions = chat_prompt_template()
    for question in list_questions:
        turn += 1
        messages = chat_template.format_messages(
            **template_params,
            chat_history=chat_history,
            input=question
        )
        response = model.invoke(messages)
        print(f"助手：{response.content}\n")

        # 更新对话历史
        chat_history.append(("human", question))
        chat_history.append(("assistant", response.content))

        # total_tokens
        tokens_info = tokens_used(response)
        total_tokens += tokens_info['total_tokens']
        print(f'本轮使用量：{tokens_info} \n 累计用量：{total_tokens}')
    return total_tokens, turn  # 注意return的层级


def tokens_used(response):
    try:
        usage = response.response_metadata.get('token_usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }
    except Exception as e:
        print(e)
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def main():
    try:
        total_tokens, turns = chat_invoke()
        print(f"对话完成！共 {turns} 轮，总Token使用: {total_tokens}")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
