# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
# 实例：构建一个简单的聊天机器

from os import getenv

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = init_chat_model(
    model='deepseek-reasoner',
    api_key=api_key,
    temperature=0.8,
)


def chat_system_template():
    system_prompt = '你是一个{feature}的{profession}。请保持{style}的回复风格。'

    chat_system_prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}')
    ])
    '''
    chat_system_template = ChatPromptTemplate.from_messages
    chat_system_template = ChatPromptTemplate
    区别？？？
    '''
    return chat_system_prompt_template


def system_messages_his(prompt_params):
    feature = prompt_params.get('feature')
    profession = prompt_params.get('profession')
    style = prompt_params.get('style')
    content = f'你是一个{feature}的{profession}。请保持{style}的回复风格。'
    return SystemMessage(content=content)


def chat_params():
    prompt_params = {
        'feature': '专业、langchain为主',
        'profession': 'AI应用开发工程师',
        'style': '简要兼具学术性',
    }
    chat_history = []

    question_list = [
        '你好！',
        # '你能做什么？简短回答',
        # '解释一下langchain中的LCEL',
        # '在上海，该岗位的入门员工薪资是多少'
    ]
    return prompt_params, chat_history, question_list


def chat_response():
    prompt_params, chat_history, question_list = chat_params()
    total_tokens = 0
    turn = 0
    chat_template = chat_system_template()
    system_messages = system_messages_his(prompt_params)

    for question in question_list:
        turn += 1
        chat_prompt_template = chat_template.format_messages(
            **prompt_params,
            chat_history=chat_history,
            input=question,
        )

        response = model.invoke(chat_prompt_template)
        print(response.content)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response.content))

        tokens_info = total_tokens_used(response)
        total_tokens += tokens_info['total_tokens']
        print(f'本轮使用量：{tokens_info} \n 累计用量：{total_tokens}')

    def system_prompt():
        system_prompt_his = chat_prompt_template[0]
        sys_his = [system_prompt_his]
        print(sys_his)

    return turn, total_tokens, chat_history, system_prompt, system_messages


def total_tokens_used(response):
    try:
        token_usage = response.response_metadata.get('token_usage', {})
        return {
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'total_tokens': token_usage.get('total_tokens', 0)
        }
    except Exception as e:
        print(e)
        return {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}


def main():
    try:
        turns, total_tokens, chat_history, system_prompt, system_messages = chat_response()
        print(f"对话完成！共 {turns} 轮，总Token使用: {total_tokens}")
        print(f'对话历史如下：{system_messages}')
        for i in chat_history:
            print(f'对话历史如下：{i}')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
