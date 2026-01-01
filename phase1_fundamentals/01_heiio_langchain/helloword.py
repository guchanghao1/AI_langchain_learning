# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------

from dotenv import load_dotenv
from os import getenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# import dotenv
# from os import environ
# from langchain_openai import ChatOpenAI

# init_chat_model方式（还有chainopenai方式）

# 加载环境变量
load_dotenv()
# dotenv.load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")
base_url = getenv("DEEPSEEK_BASE_URL")
# 带上/v1可以输出,不带则显示超时???亦可在初始化模型中不加入base_url参数!!为什么??
# 其他写法
'''
environ['DEEPSEEK_API_KEY'] = getenv("DEEPSEEK_API_KEY")
environ['DEEPSEEK_BASE_URL']= getenv("DEEPSEEK_BASE_URL")
冗余操作：从环境变量读取，再写回环境变量（无意义）
潜在覆盖：如果环境变量本来就存在，这样做可能改变原有的值
逻辑混乱：如果环境变量不存在，会写入 None 或空值
调试困难：难以追踪环境变量的变化来源
违反原则：没有明显的好处却增加了复杂度
'''

if not api_key:
    raise ValueError("API key is required")

# 初始化模型 - 添加参数
model = init_chat_model(
    model="deepseek-chat",
    base_url=base_url,
    api_key=api_key,
)


# 1.简单的LLM调用
# 使用 invoke 方法
def example_1_simple_invoke():
    response = model.invoke('人工智能应用是什么')
    print(response)
    print(f'返回对象类型:{type(response)}')
    # 返回对象类型: <class 'langchain_core.messages.ai.AIMessage'>
    print(f'AI回复:{response.content}')


# 2.Message列表
'''
核心概念：
SystemMessage: 系统消息，用于设定 AI 的行为和角色
HumanMessage: 用户消息
AIMessage: AI 的回复消息
'''


def example_2_massage_invoke():
    messages = [
        SystemMessage(content='你是一个AI应用开发就业导师,回答字数不超过50字。'),
        HumanMessage(content='2026年AI应用开发岗位有哪些,请简单列举.'),
    ]
    ai_rep1 = model.invoke(messages)

    print('系统提示:' + messages[0].content)
    print('用户提问:' + messages[1].content)
    print(ai_rep1.content)

    messages.append(ai_rep1)
    hum_2 = HumanMessage(content='作为一个小白转行,学习三个月你推荐选择什么岗位.')
    messages.append(hum_2)
    print('用户提问:' + messages[-1].content)

    ai_rep2 = model.invoke(messages)
    print(ai_rep2.content)


# 用字典格式传递消息 (推荐)
def example_3_dict_invoke():
    messages = [
        {'role': 'system', 'content': '你是一个AI应用开发就业导师,回答字数不超过50字。'},
        {'role': 'user', 'content': '2026年以后十年的AI应用开发的前景.'},

    ]
    for mes in messages:
        print(f'{mes["role"]}:{mes["content"]}')

    response = model.invoke(messages)
    print(response.content)

    messages.append(response)
    hum_dict = {'role': 'user', 'content': '有简单的课程实战项目,无落地经验,在上海能找到岗新多少的ai应用工作?'}
    messages.append(hum_dict)
    print(f'{hum_dict["role"]}:{hum_dict["content"]}')

    response2 = model.invoke(messages)
    print(response2.content)


# 配置模型参数
def example_4_model_parameters():
    tem_lst = [0.2, 1, 1.8]
    for i in tem_lst:
        model_deterministic = init_chat_model(
            model="deepseek-chat",
            temperature=i,
            max_tokens=50,
            timeout=10.0,
        )

        messages = [
            {'role': 'system', 'content': '你是一名AI应用开发导师。'},
            {'role': 'user', 'content': '给我一个AI应用学习项目建议。'}
        ]

        for mes in messages:
            print(f'{mes["role"]}:{mes["content"]}')

        response = model_deterministic.invoke(messages)
        print(response.content)
    # 参数temperature与timeout
    '''
    temperature 是一个介于 0.0 到 2.0 之间的浮点数，它控制模型输出的随机性程度：
    Temperature 值	含义	适用场景
    0.0	完全确定性，每次输出相同	代码生成、数学计算
    0.1-0.5	较低随机性，比较稳定	事实问答、技术文档
    0.5-0.8	中等随机性，平衡稳定与创意	通用对话、内容创作
    0.8-1.2	较高随机性，比较有创意	创意写作、头脑风暴
    1.2-2.0	高随机性，可能不太连贯	实验性创作、诗歌

    timeout 是一个浮点数（单位：秒），表示：
    等待 API 响应的最长时间
    超过这个时间就自动取消请求
    '''


# invoke方法的返回值
def example_5_response_structure():
    model_response = init_chat_model(
        model="deepseek-chat",
        temperature=1.8,
        max_tokens=200,
        timeout=10.0,
    )
    response = model_response.invoke('给我一首完整的励志诗。')
    print(response.content)
    print(response.type)
    print(response.id)


# 异常处理
def example_6_error_handing():
    try:
        model_error = init_chat_model(
            model="deepseek-chat",
        )
        res = model_error.invoke('hello')
        print(res.content)
    except ValueError as v:
        print(v)
    except ConnectionError as ce:
        print(ce)
    except Exception as ex:
        print(type.__name__, ex)


def example_7_models_text():
    model_list = ['deepseek-chat', 'deepseek-reasoner']
    for model in model_list:
        try:
            model_text = init_chat_model(
                model=model,
            )
            res = model_text.invoke('hello')
            print(f'模型{model}:{res.content}')
        except Exception as ex:
            print(ex)


def main():
    try:
        # example_1_simple_invoke()
        # example_2_massage_invoke()
        # example_3_dict_invoke()
        # example_4_model_parameters()
        example_5_response_structure()
        # example_6_error_handing()
        # example_7_models_text()
    except Exception as ex:
        print(ex)
        # 打印所有的报错信息
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
