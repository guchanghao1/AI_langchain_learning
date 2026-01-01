"""
LangChain 1.0 - Simple Agent (使用 create_agent)
=====================================================

本模块重点讲解：
1. 使用 create_agent 创建 Agent（LangChain 1.0 新 API）
2. Agent 自动决定何时使用工具
3. Agent 执行循环的工作原理

⚠️ 重要更新：
- LangChain 1.0 中，Agent 创建使用 `create_agent`
- 它来自 `langchain.agents` 模块（LangChain 1.0 新增）
- 旧的 `create_react_agent`（langgraph.prebuilt）已弃用
"""

from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # ✅ LangChain 1.0 API
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver  # 用于多轮对话

# 导入自定义工具
from tools.weather import get_weather
from tools.calculator import calculator
from tools.web_search import web_search

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.5,
)


# 示例 1：创建第一个 Agent
def example_1_basic_agent():
    """
    示例1：创建最简单的 Agent
    关键：
    1. 使用 create_agent 函数（LangChain 1.0 API）
    2. 传入 model 和 tools
    3. Agent 会自动决定是否使用工具
    """

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt='你是一个有帮助的助手。'
    )

    # 测试需要工具的问题
    response1 = agent.invoke({
        "messages": [{"role": "human", "content": "北京今天天气怎么样？"}]  # 关于invoke的输入输出
    })
    print(response1["messages"][-1].content)  # 关于agent的输出

    # 测试不需要工具的问题
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "你好，介绍一下你自己"}]
    })
    print(response2["messages"][-1].content)


# 示例 2：多工具 Agent
def example_2_multi_tool_agent():
    """
       示例2：配置多个工具的 Agent

       Agent 会根据问题选择合适的工具
       """
    print("\n" + "=" * 70)
    print("示例 2：多工具 Agent")
    print("=" * 70)

    # 创建配置多个工具的 Agent
    agent = create_agent(
        model=model,
        tools=[get_weather, calculator, web_search],
        system_prompt="你是一个有帮助的助手。"
    )

    print("\n配置的工具：")
    print("  - get_weather（天气查询）")
    print("  - calculator（计算器）")
    print("  - web_search（网页搜索）")

    # 测试不同类型的问题
    tests = [
        "上海的天气怎么样？",  # 应该用 get_weather
        "15 乘以 23 等于多少？",  # 应该用 calculator
    ]

    for i, question in enumerate(tests, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}：{question}")
        print(f"{'=' * 70}")

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        # 显示最终回答
        print(f"\nAgent 回复：{response['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 从多个工具中选择最合适的")
    print("  - 基于工具的 docstring 理解工具用途")


# 示例 3：带系统提示的 Agent
def example_3_agent_with_system_prompt():
    """
    示例3：自定义 Agent 的行为

    使用 prompt 参数（注意：不是 system_prompt）
    """
    print("\n" + "=" * 70)
    print("示例 3：自定义 Agent 行为")
    print("=" * 70)

    # create_agent 使用 system_prompt 参数（字符串或 SystemMessage）
    system_message = SystemMessage(
        content=
        """
        你是一个友好的助手。
        特点：
        - 回答简洁明了
        - 使用工具前先说明
        - 结果用表格或列表清晰展示
        """)

    agent = create_agent(
        model=model,
        tools=[get_weather, calculator],
        system_prompt=system_message  # ✅ 使用 system_prompt 参数
    )

    print("\n测试：自定义行为的 Agent")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何？顺便算一下 100 加 50"}]
    })

    print(f"\nAgent 回复：{response['messages'][-1].content}")

    print("\n关键点：")
    print("  - system_prompt 参数定义 Agent 的系统提示")
    print("  - 可以指定输出格式、语气、工作流程等")
    print("  - 也可以传入 SystemMessage 对象")


# 示例 4：Agent 执行过程详解
def example_4_agent_execution_details():
    """
    示例4：查看 Agent 执行的完整过程

    理解 Agent 如何一步步工作
    """
    print("\n" + "=" * 70)
    print("示例 4：Agent 执行过程详解")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：25 乘以 8 等于多少？")
    print("\nAgent 执行过程：")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
    })

    print('完整信息历史')
    for i, mesg in enumerate(response['messages'], 1):
        print(f'消息{i}--{mesg.__class__.__name__}')
        if hasattr(mesg, 'content'):
            print(f'内容：{mesg.content}')
        if hasattr(mesg, 'tool_calls') and mesg.tool_calls:
            print(f'工具：{mesg.tool_calls}')

    print("\n执行循环：")
    print("""
        1. 用户提问 → HumanMessage
        2. AI 决定调用工具 → AIMessage (包含 tool_calls)
        3. 执行工具 → ToolMessage (包含结果)
        4. AI 基于结果生成答案 → AIMessage (最终回答)
        """)


# 示例 5：多轮对话 Agent（使用 MemorySaver）
def example_5_memory_saver():
    memory = InMemorySaver()

    # 创建配置多个工具的 Agent
    agent = create_agent(
        model=model,
        tools=[get_weather, calculator, web_search],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=memory,
        # checkpointmemory=memory,create_agent() got an unexpected keyword argument 'checkpointmemory'
    )
    # 使用 thread_id 来保持对话
    config = {'configurable': {'thread_id': 'text_one'}}

    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "今天上海天气怎么样？"}]},
        config=config
    )

    print(response1["messages"][-1].content)

    # 第二轮：继续上一轮的对话（记忆自动保持）
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "介绍一下这个城市。"}]},
        config=config
    )
    print(f"Agent：{response2['messages'][-1].content}")

    print("\n关键点：")
    print("  - 使用 MemorySaver 作为 checkpointer")
    print("  - 通过 thread_id 区分不同的对话")
    print("  - Agent 自动记住上下文")
    print("  - 不需要手动传递历史消息")


def main():
    try:
        example_1_basic_agent()
        example_2_multi_tool_agent()
        example_3_agent_with_system_prompt()
        example_4_agent_execution_details()
        example_5_memory_saver()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
