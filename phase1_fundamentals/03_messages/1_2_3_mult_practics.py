# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------

# 实例：构建一个简单的聊天机器人(1.加入多套对话模板；2.保留最近N组对话；3.学习LCEL基础，初步应用)
''' 思路：构建chain链--动态化系统提示--预备各种参数--调用模型获取响应--保存最近N轮对话--监控tokens--简单的异常处理'''

from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.7,
)


def chat_template_and_chain():
    system_template = '你是一个{feature}的{profession}。请保持{style}的回复风格。'

    chat_sys_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        MessagesPlaceholder(variable_name='chat_history_list'),
        ('human', '{input}'),
    ])

    def prepare_params(data):
        return {
            'feature': data.get('feature'),
            'profession': data.get('profession'),
            'style': data.get('style'),
            'chat_history_list': data.get('chat_history_list'),
            'input': data.get('input'),
        }

    chain = (
            RunnablePassthrough()
            | prepare_params
            | chat_sys_template
            | model
    )

    return chain


def chat_params():
    # 对于ChatPromptTemplate的输入与输出理解需要加强
    prompt_params = {
        'feature': '专业、langchain为主',
        'profession': 'AI应用开发工程师',
        'style': '简要兼具学术性',
    }

    chat_history_list = []

    question_list = [
        '介绍一下langchain表达式'
        'LCEL的核心是什么',
        '，具体分为哪几部分？',
        '实际开发中要学习到什么水平？'
    ]
    return prompt_params, chat_history_list, question_list


def chat_response():
    total_used = 0
    chain = chat_template_and_chain()
    prompt_params, chat_history_list, question_list = chat_params()

    for i, question in enumerate(question_list, 1):
        print(f"第{i}轮对话: {question}")
        input_list = {
            **prompt_params,
            'chat_history_list': chat_history_list,
            'input': question,
        }

        response = chain.invoke(input_list)
        print(f'AI回复：{response.content}')

        chat_history_list.append(HumanMessage(content=question))
        chat_history_list.append(AIMessage(content=response.content))

        tokens_used, total_used = tokens_total_used(response, total_used)
        print(f'本轮Token使用量：{tokens_used["total_tokens"]}')
    return total_used, chat_history_list


def tokens_total_used(response, total_used):
    tokens_used = response.response_metadata.get('token_usage', {})
    total_used += tokens_used.get('total_tokens', 0)
    return tokens_used, total_used


def main():
    total_used, chat_history_list = chat_response()
    print(f"对话完成！，总Token使用: {total_used}")
    return chat_history_list


if __name__ == '__main__':
    main()
