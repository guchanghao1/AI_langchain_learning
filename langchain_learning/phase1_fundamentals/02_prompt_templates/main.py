# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

api_key = getenv('DEEPSEEK_API_KEY')

model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    max_tokens=500,
)


# 1.为什么用提示词模板（对比字符串拼接）
def example_1_why_template():
    # 字符串拼接
    topic = 'AI应用开发'
    difficulty = 'medium'
    prompt_str = f'你是一个{difficulty}级别的编程导师。请用简单易懂的语言解释{topic}。'
    response_str = model.invoke(prompt_str)
    print(response_str.content)

    # 简单提示词模板 PromptTemplate
    template = PromptTemplate.from_template(
        '你是一个{difficulty}级别的编程导师，请用简单易懂的语言解释{topic}。'
    )
    prompt = template.format(
        difficulty=difficulty,
        topic=topic,
    )
    response_prompt = model.invoke(prompt)
    print(response_prompt.content)
    '''💡 优势：
    1. 可复用 - 同一个模板可以用于不同的输入
    2. 可维护 - 模板和数据分离，易于修改
    3. 类型安全 - 自动验证变量
    4. 可测试 - 更容易编写测试用例'''


# 2：PromptTemplate 基础用法
def example_2_prompt_template_basics():
    # PromptTemplate用于简单场景

    # 1.from_template（最推荐）

    template_first = PromptTemplate.from_template(
        '将下列文本翻译成{language}：\n{text}'
    )
    prompt_first = template_first.format(
        language='韩语',
        text='你好，我是AI应用工程师。'
    )
    print(prompt_first)

    res_first = model.invoke(prompt_first)
    print(res_first.content)

    # 2.显示指定变量（语法更严格）
    template_second = PromptTemplate(
        input_variables=['product', 'feature'],
        template='为{product}产品编写一段广告标语，以{feature}为核心,以{language}显示。',
    )
    prompt_second = template_second.format(
        product='智能手表',
        feature='掌控时间',
        language='中文繁体'
    )
    print(prompt_second)

    res_second = model.invoke(prompt_second)
    print(res_second.content)
    print(res_second)

    # 3.invoke直接生成（更方便）
    template_third = PromptTemplate.from_template(
        "为{season}写一首{style}诗，字数{count}"
    )

    # invoke 直接返回格式化后的值
    prompt_third = template_third.invoke({
        'season': '冬天',
        'style': '现代',
        'count': '100',
    })
    print(prompt_third)

    res_third = model.invoke(prompt_third)
    print(res_third.content)


# 3.ChatPromptTemplate 聊天消息模板
def example_3_chatprompttemplate():
    # 使用元组格式（最简单，推荐）
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{expertise}。"),
        ("user", "请给我{task}"),
    ])

    messages = template.format_messages(
        role='AI教授',
        expertise='大模型开发',
        task='解释什么是机器学习',
    )
    print(messages)

    res_chat = model.invoke(messages)
    print(res_chat.content)


# “更细粒度的控制”指的是能够深入到框架的各个组件和流程中，进行定制化调整和干预。
def main():
    try:
        # example_1_why_template()
        # example_2_prompt_template_basics()
        example_3_chatprompttemplate()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
