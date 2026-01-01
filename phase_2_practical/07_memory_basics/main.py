# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------
"""
LangChain 1.0 - Memory Basics (内存管理基础)
==========================================
本模块重点讲解：
1. InMemorySaver - LangGraph 提供的内存管理
2. checkpointer 参数 - 为 Agent 添加内存
3. thread_id - 会话管理
4. 多轮对话状态保持
"""

from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

# from langchain_openai import ChatOpenAI

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.6
)


# 创建一个简单的工具
@tool
def get_used_info(used_id: str) -> str:
    """获取简单的用户消息"""
    users = {
        "123": "张三，25岁，工程师",
        "456": "李四，30岁，设计师"
    }
    return users.get(used_id, '未找到该用户')


def example_1_with_memory():
    agent = create_agent(
        model=model,
        system_prompt='你是一位大模型开发工程师。',
        checkpointer=InMemorySaver()
    )

    config = {'configurable': {'thread_id': 'text_one'}}
    message1 = [{'role': 'human', 'content': '简单介绍一下机器学习'}]
    response = agent.invoke({'messages': message1}, config=config)
    # print(response)#{'messages': [HumanMessage(content='简单介绍一下机器学习', additional_kwargs={}, response_metadata={}, id='d4971dfd-1373-482b-95ec-bf51744790d6'), AIMessage(content='机器学习是人工智能（AI）的核心分支，专注于让计算机系统通过数据自动学习和改进，而无需显式编程。其核心思想是从历史数据中识别模式，并利用这些模式对新数据进行预测或决策。\n\n### 核心概念：\n1. **数据驱动**：模型性能高度依赖于数据的质量和数量。\n2. **训练与预测**：通过算法分析训练数据构建模型，再对未知数据做出预测。\n3. **泛化能力**：模型处理新数据的能力，避免“过拟合”（只记忆训练数据）或“欠拟合”（未能捕捉数据规律）。\n\n### 主要类型：\n- **监督学习**：使用带标签的数据训练模型，用于分类（如图像识别）和回归（如房价预测）。\n- **无监督学习**：从无标签数据中发现隐藏结构，如聚类（客户分群）和降维。\n- **强化学习**：智能体通过与环境互动学习最优策略，如游戏AI和机器人控制。\n- **深度学习**：基于神经网络的复杂模型，擅长处理图像、语音等高维数据。\n\n### 关键步骤：\n1. 数据收集与清洗\n2. 特征工程（提取关键信息）\n3. 模型选择与训练\n4. 评估与优化（如交叉验证）\n5. 部署与监控\n\n### 应用领域：\n- 自然语言处理（如ChatGPT）\n- 计算机视觉（如人脸识别）\n- 推荐系统（如电商平台）\n- 自动驾驶、医疗诊断等\n\n机器学习正推动各行业智能化，但也面临数据偏见、可解释性等挑战。随着算力和算法进步，它将继续扩展人类能力的边界。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 343, 'prompt_tokens': 14, 'total_tokens': 357, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 14}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '7d5e4f6e-f78c-4c47-b822-3aca3b24343c', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b7200-9881-7900-aa14-fb3532cadf32-0', usage_metadata={'input_tokens': 14, 'output_tokens': 343, 'total_tokens': 357, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}

    # 第二轮
    message2 = [{'role': 'human', 'content': '简单说明上述内容与深度学习的关系。'}]
    response2 = agent.invoke({'messages': message2}, config=config)
    print(response2[
              'messages'])  # [HumanMessage(content='简单介绍一下机器学习', additional_kwargs={}, response_metadata={}, id='9cab75d0-b05a-49d6-8fd8-e1dc6f5b155d'), AIMessage(content='好的，机器学习是人工智能的一个核心分支，其核心思想是：**让计算机系统能够从数据中“学习”规律和模式，并利用这些学习到的知识对新的、未见过的数据做出预测或决策，而无需为每个特定任务进行明确的、手动的编程。**\n\n简单来说，就是**教计算机从经验中学习**。\n\n### 核心概念类比\n\n想象一下教一个孩子识别猫：\n*   **传统编程**：你需要手动编写一个极其复杂的规则列表（“有胡须、尖耳朵、长尾巴、喵喵叫...”），程序会逐条检查。如果遇到无毛猫，可能就识别不了。\n*   **机器学习**：你给孩子看成千上万张猫和不是猫的图片，并告诉他哪些是猫。孩子的大脑（算法）会自己找出区分猫的关键特征（模式）。之后，当他看到一张新图片时，即使是从未见过的品种，他也能大概率判断出是不是猫。\n\n### 机器学习的三大主要类型\n\n1.  **监督学习**\n    *   **核心**：使用 **“有标签”** 的数据进行训练。数据集中每个样本都包含了输入数据和对应的正确答案（标签）。\n    *   **目标**：学习输入到输出之间的映射关系，以便对新的输入数据做出预测。\n    *   **典型任务**：\n        *   **分类**：预测离散类别。例如：垃圾邮件过滤（是/否）、图像识别（猫/狗/汽车）、疾病诊断（患病/健康）。\n        *   **回归**：预测连续数值。例如：预测房价、预测股票价格、预测销售额。\n\n2.  **无监督学习**\n    *   **核心**：使用 **“无标签”** 的数据进行训练。数据集中只有输入数据，没有给定的答案。\n    *   **目标**：发现数据中隐藏的内在结构、模式或分组。\n    *   **典型任务**：\n        *   **聚类**：将数据自动分组。例如：客户细分、社交网络分析、新闻主题分组。\n        *   **降维**：在保留主要信息的前提下，减少数据的特征数量，便于可视化或后续处理。\n\n3.  **强化学习**\n    *   **核心**：智能体通过与环境的**交互**和**试错**来学习。它通过执行动作、获得奖励或惩罚，来学习在特定环境下采取何种行动能使长期累积奖励最大化。\n    *   **类比**：训练一只狗做动作，做对了就给零食（正奖励），做错了就无视或轻微制止（负奖励）。\n    *   **典型应用**：AlphaGo（围棋）、机器人控制、自动驾驶决策、游戏AI。\n\n### 基本工作流程\n\n一个典型的机器学习项目通常包含以下步骤：\n1.  **数据收集与准备**：获取原始数据，并进行清洗（处理缺失值、异常值）、格式化。\n2.  **特征工程**：从原始数据中提取或构造对预测任务有用的特征（变量）。这是非常关键且需要经验的一步。\n3.  **模型选择**：根据问题类型（分类、回归等）和数据特点，选择合适的算法（如决策树、神经网络、支持向量机等）。\n4.  **模型训练**：使用准备好的数据，让算法“学习”数据中的模式。\n5.  **模型评估**：使用模型未见过的测试数据来评估其性能（如准确率、精确率、召回率等）。\n6.  **超参数调优与优化**：调整模型的设置（超参数），以提升性能。\n7.  **预测与部署**：将训练好的模型应用到新的真实数据中，并集成到产品或服务中。\n\n### 为什么机器学习如此重要？\n\n*   **处理复杂问题**：对于规则难以描述的问题（如图像识别、自然语言理解），机器学习表现出色。\n*   **从海量数据中挖掘价值**：在大数据时代，机器学习是分析和利用数据的关键工具。\n*   **自动化与智能化**：能够自动化执行复杂的预测和决策任务，提升效率。\n\n### 与深度学习的关系\n\n**深度学习是机器学习的一个子领域**，它主要使用被称为“人工神经网络”的复杂模型，尤其是具有很多隐藏层的“深度神经网络”。深度学习在图像、语音、文本等非结构化数据处理方面取得了革命性突破，是当前人工智能浪潮的主要驱动力。\n\n**总结一下**：机器学习是一门赋予计算机从数据中学习能力的科学。它正在深刻地改变我们生活的方方面面，从手机上的语音助手、推荐系统，到医疗诊断和自动驾驶汽车，背后都有它的身影。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 946, 'prompt_tokens': 14, 'total_tokens': 960, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 14}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '5ba56161-4dec-4761-ac00-952aa940268e', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b7208-21a0-7ef1-bc23-e0155952ca0f-0', usage_metadata={'input_tokens': 14, 'output_tokens': 946, 'total_tokens': 960, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}), HumanMessage(content='简单说明上述内容与深度学习的关系。', additional_kwargs={}, response_metadata={}, id='fc891fce-ecf7-4651-a6ec-5f549514f1fa'), AIMessage(content='好的，用一个简单清晰的比喻和说明来解释机器学习与深度学习的关系：\n\n### 核心关系：**深度学习是机器学习的一个重要的、强大的子集。**\n\n你可以把它们想象成一种 **“包含”** 关系：\n\n*   **机器学习** 是一个广阔的领域，包含了许多让计算机从数据中学习的方法。\n*   **深度学习** 是机器学习这个“工具箱”里的一套**特别强大且当前最流行的工具**。\n\n---\n\n### 关键区别与联系\n\n1.  **模型结构：从“手工特征”到“自动学习”**\n    *   **传统机器学习**：非常依赖 **“特征工程”**。专家需要手动从原始数据中提取和设计出对预测有用的特征（例如，从图片中提取颜色、纹理、形状等统计量）。模型学习的是这些**人工设计的特征**之间的关系。\n    *   **深度学习**：使用 **“深度神经网络”**。它像一个多层的、自动化的特征提取和决策流水线。你只需要输入最原始的数据（如图像的原始像素、文本的单词），神经网络的前几层会自动学习并提取出**从简单到复杂的特征**（例如，从像素到边缘，到局部形状，再到整个物体）。它**自动完成了特征工程**的核心部分。\n\n2.  **数据与性能**\n    *   **传统机器学习**：在数据量较小、特征设计良好的情况下，表现优异且效率高。\n    *   **深度学习**：通常需要**海量的数据**（如数百万张图片）和强大的计算力（如GPU）。但当数据量足够大时，它在处理**非结构化数据**（图像、声音、文本）的复杂任务上，性能往往能远超传统方法。\n\n3.  **可解释性**\n    *   **传统机器学习**：模型（如决策树、线性回归）通常更容易理解和解释，我们可以知道模型是如何做出判断的。\n    *   **深度学习**：模型结构复杂，像一个“黑箱”。我们很难精确理解网络内部具体是如何做出某个决策的，这是其面临的主要挑战之一。\n\n---\n\n### 一个直观的类比：识别手写数字\n\n*   **用传统机器学习方法**：\n    1.  专家需要设计特征：这个数字有多少个闭合圆圈？笔画的交叉点有几个？垂直和水平线条的比例是多少？\n    2.  将这些手工计算的特征输入给一个分类器（如支持向量机SVM）进行学习。\n\n*   **用深度学习方法**：\n    1.  直接将手写数字的**原始像素图片**输入到一个深度神经网络（如卷积神经网络CNN）。\n    2.  网络的第一层可能学会识别“边缘”，第二层学会组合边缘成“拐角”和“短线条”，第三层学会组合成“数字的局部结构”（如一个圈或一条竖线），最后一层判断这是哪个数字。\n    3.  **整个过程是端到端自动学习的**。\n\n### 总结\n\n| 方面 | 机器学习 | 深度学习 |\n| :--- | :--- | :--- |\n| **关系** | **总领域** | **子领域，一种特定方法** |\n| **核心** | 从数据中学习的各种算法 | 基于深层神经网络的算法 |\n| **特征处理** | 主要依赖**人工特征工程** | **自动学习特征层次** |\n| **数据需求** | 可大可小 | 通常需要**海量数据** |\n| **擅长领域** | 结构化数据、中小规模任务 | **非结构化数据**（图像、语音、文本）、复杂模式 |\n\n**简单来说：**\n**机器学习** 是让计算机学习的总科学。**深度学习** 是使用一种名为“深度神经网络”的特定模型来实现机器学习，这种方法在处理像看、听、读这类人类直觉擅长的事情上，效果尤其惊人，是当前人工智能突破的主要引擎。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 796, 'prompt_tokens': 972, 'total_tokens': 1768, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 128}, 'prompt_cache_hit_tokens': 128, 'prompt_cache_miss_tokens': 844}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '1b070912-b4e1-4de0-a597-74bcabf5e1fb', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b7208-a5f4-7fb1-8af2-95892cb7709a-0', usage_metadata={'input_tokens': 972, 'output_tokens': 796, 'total_tokens': 1768, 'input_token_details': {'cache_read': 128}, 'output_token_details': {}})]


def example_2_multiple_threads():
    """
    示例2：管理多个独立的会话
    关键：不同的 thread_id = 不同的对话
    """
    print("\n" + "=" * 70)
    print("示例 3：多个独立会话")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver()
    )

    # 会话 1
    config1 = {"configurable": {"thread_id": "user_alice"}}
    print("\n[会话 1 - Alice]")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我叫 Alice"}]},
        config=config1
    )
    print("Alice: 我叫 Alice")

    # 会话 2
    config2 = {"configurable": {"thread_id": "user_bob"}}
    print("\n[会话 2 - Bob]")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我叫 Bob"}]},
        config=config2
    )
    print("Bob: 我叫 Bob")

    # 回到会话 1
    print("\n[回到会话 1 - Alice]")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        config=config1
    )
    print(f"Agent: {response1['messages'][-1].content}")

    # 回到会话 2
    print("\n[回到会话 2 - Bob]")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        config=config2
    )
    print(
        f"Agent: {response2}")  # Agent: {'messages': [HumanMessage(content='我叫 Bob', additional_kwargs={}, response_metadata={}, id='7b812405-286b-4999-9ad0-50972c107f37'), AIMessage(content='你好 Bob！很高兴认识你！有什么我可以帮助你的吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 12, 'total_tokens': 25, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 12}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '6b57409b-a31c-46cb-9102-e06d4c9c32db', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b720b-b295-7603-a2c6-b612f5bcb8db-0', usage_metadata={'input_tokens': 12, 'output_tokens': 13, 'total_tokens': 25, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}), HumanMessage(content='我叫什么？', additional_kwargs={}, response_metadata={}, id='f671a55c-377e-4c74-bdad-4d30457f7f14'), AIMessage(content='你刚才告诉我你的名字是 **Bob**。如果这不是你真实的名字，或者你想让我称呼你为其他名字，请随时告诉我！ 😊', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 32, 'total_tokens': 63, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 32}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '5aa98426-ef98-44e7-a07a-b42ac36efb14', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b720b-bdc2-7853-b712-ba45dbdb5321-0', usage_metadata={'input_tokens': 32, 'output_tokens': 31, 'total_tokens': 63, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}

    print("\n关键点：")
    print("  - 不同 thread_id 的会话完全独立")
    print("  - Agent 能正确记住每个会话的内容")
    print("  - 适合多用户聊天场景")


def example_3_practical_use():
    """
    简单的实际场景应用
    模拟一个记住用户消息的客服场景
    """
    system_prompt = """
    你是一个客服助手。
    特点：
    - 记住用户说过的话
    - 友好、有耐心
    - 使用 get_user_info 工具查询用户信息时需要用户 ID"""

    agent = create_agent(
        model=model,
        tools=[get_used_info],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

    config = {'configurable': {'thread_id': '客服1号'}}
    conversations = [
        "你好，我想咨询一下",
        "我的用户 ID 是 123",
        "帮我查一下我的信息",
        "我多大来着？"  # 测试记忆
    ]
    for conversation in conversations:
        print(f"用户：{conversation}")
        messages = {'role': 'human', 'content': conversation}
        response = agent.stream({'messages': [messages]}, config=config)
        for chunk in response:
            print(chunk)


# 流式输出如下
"""
用户：你好，我想咨询一下
{'model': {'messages': [AIMessage(content='您好！很高兴为您服务。我是客服助手，有什么可以帮助您的吗？\n\n请问您想咨询什么方面的问题呢？比如账户信息、服务使用、或者其他任何需要帮助的事情，我都很乐意为您解答。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 44, 'prompt_tokens': 346, 'total_tokens': 390, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 26}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': 'e304a0c2-c5f1-448f-adb0-d7646b111498', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b722a-dc55-7741-b493-b5d6a27b9677-0', usage_metadata={'input_tokens': 346, 'output_tokens': 44, 'total_tokens': 390, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}})]}}
用户：我的用户 ID 是 123
{'model': {'messages': [AIMessage(content='好的，我记下了您的用户ID是123。现在有什么可以帮您的吗？\n\n如果您需要查询您的账户信息，我可以帮您查看。或者您有其他任何问题，请随时告诉我。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 40, 'prompt_tokens': 401, 'total_tokens': 441, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 17}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '175fb87a-3bd4-4446-883b-630373a4223b', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b722a-e75e-73d3-b666-2fd052f76294-0', usage_metadata={'input_tokens': 401, 'output_tokens': 40, 'total_tokens': 441, 'input_token_details': {'cache_read': 384}, 'output_token_details': {}})]}}
用户：帮我查一下我的信息
{'model': {'messages': [AIMessage(content='好的，我来帮您查询一下您的用户信息。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 56, 'prompt_tokens': 450, 'total_tokens': 506, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 66}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '61a8a9ef-943b-41cf-9a5a-ac64309136a2', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019b722a-f1b5-7e91-a65a-afd5bbc6306f-0', tool_calls=[{'name': 'get_used_info', 'args': {'used_id': '123'}, 'id': 'call_00_2UF1TJzYAcIe3TxFP6R3QT45', 'type': 'tool_call'}], usage_metadata={'input_tokens': 450, 'output_tokens': 56, 'total_tokens': 506, 'input_token_details': {'cache_read': 384}, 'output_token_details': {}})]}}
{'tools': {'messages': [ToolMessage(content='张三，25岁，工程师', name='get_used_info', id='cd11d250-2279-4578-91e2-2037532db6e0', tool_call_id='call_00_2UF1TJzYAcIe3TxFP6R3QT45')]}}
{'model': {'messages': [AIMessage(content='根据查询结果，您的用户信息如下：\n\n- **姓名**：张三\n- **年龄**：25岁\n- **职业**：工程师\n\n这些信息正确吗？如果您需要更新信息或有其他问题，请告诉我，我会尽力帮助您。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 52, 'prompt_tokens': 529, 'total_tokens': 581, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 448}, 'prompt_cache_hit_tokens': 448, 'prompt_cache_miss_tokens': 81}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '02821b84-2a65-4851-a58e-117517f2238e', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b722a-fc81-7da2-a12f-5b4c675d76be-0', usage_metadata={'input_tokens': 529, 'output_tokens': 52, 'total_tokens': 581, 'input_token_details': {'cache_read': 448}, 'output_token_details': {}})]}}
用户：我多大来着？
{'model': {'messages': [AIMessage(content='根据刚才查询到的信息，您是**25岁**。\n\n如果您需要确认或更新年龄信息，我可以帮您进一步处理。还有其他问题吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 589, 'total_tokens': 619, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 576}, 'prompt_cache_hit_tokens': 576, 'prompt_cache_miss_tokens': 13}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '9e3abc92-e1cb-4528-a4fc-54caa1e1dfa1', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b722b-0887-7a22-881f-a75beb92faef-0', usage_metadata={'input_tokens': 589, 'output_tokens': 30, 'total_tokens': 619, 'input_token_details': {'cache_read': 576}, 'output_token_details': {}})]}}
"""


def main():
    try:
        example_1_with_memory()
        example_2_multiple_threads()
        example_3_practical_use()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
