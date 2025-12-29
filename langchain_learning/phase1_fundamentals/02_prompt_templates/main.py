# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()
base_url = getenv("DEEPSEEK_BASE_URL")
api_key = getenv('DEEPSEEK_API_KEY')

model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    max_tokens=50,
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


# 多轮对话模板
def example_4_conversation_template():
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。{instruction}"),
        ("user", "{question1}"),
        ("assistant", "{answer1}"),  # 目前没看出来怎么使用？？？
        ("user", "{question2}")
    ])

    # 填充模板
    messages = template.format_messages(
        role="Python 专家",
        instruction="回答要简洁、准确",
        question1="什么是列表？",
        answer1="列表是 Python 中的有序可变集合，用方括号 [] 表示。",
        question2="它和元组有什么区别？"  # 基于上下文的问题
    )

    response = model.invoke(messages)
    print(f"\nAI 回复：{response.content}\n")


# 使用MessagePromptTemplate类（高级用法）

# “更细粒度的控制”指的是能够深入到框架的各个组件和流程中，进行定制化调整和干预。

def example_5_message_prompt_template():
    system_template = SystemMessagePromptTemplate.from_template(
        '你是一个{role}，{instruction}'
    )
    human_template = HumanMessagePromptTemplate.from_template(
        '关于{topic}，我想知道{question}'
    )

    chat_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template,
    ])

    chat_prompt = chat_template.format_messages(
        role='Python老师',
        instruction='回答通俗易懂',
        topic='装饰器',
        question='原理与用法',
    )

    model_reasoner = init_chat_model(
        model='deepseek-reasoner',
        base_url=base_url,
        api_key=api_key,
        temperature=1.0,
        max_tokens=100,
        timeout=None,  # 总是超时
    )

    response_message = model_reasoner.invoke(chat_prompt)
    print(response_message.content)
    for data in response_message:
        print(data)


# 部分变量（Partial Variable）
def example_6_partial_variable():
    original_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{expertise}。"),
        ("user", "请给我{task}，概括回答"),
    ])
    print(original_template)

    partially_filled = original_template.partial(
        role='LangChain学生',
        expertise='AI app开发',
    )
    print(partially_filled)

    message_one = partially_filled.format_messages(
        task='介绍用langchain进行AI应用开发的步骤',
    )
    print(message_one)

    res_one = model.invoke(message_one)
    print(res_one.content)

    # 复用模板，不同的 task
    message_two = partially_filled.format_messages(
        task='从2026年以后的发展前景怎么样'
    )
    res_two = model.invoke(message_two)
    print(res_two.content)


# 与LCEL链式调用（预览）
def example_7_lcel_chains():
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("user", "{input}"),
    ])

    chain = template | model

    res = chain.invoke({
        "role": "幽默的程序员",
        "input": "解释什么是bug"
    })
    print(res.content)


def main():
    try:
        example_1_why_template()
        example_2_prompt_template_basics()
        example_3_chatprompttemplate()
        example_4_conversation_template()
        example_5_message_prompt_template()
        example_6_partial_variable()
        example_7_lcel_chains()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
