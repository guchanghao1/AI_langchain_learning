from os import getenv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from tools.calculator import calculator
from tools.weather import get_weather

load_dotenv()
api_key = getenv("DEEPSEEK_API_KEY")


# 初始化模型
model = init_chat_model(
    model='deepseek-chat',
    api_key=api_key,
    temperature=0.5,
    streaming=True,
)


def example_2_streaming():
    """
    示例2：实时查看 Agent 的输出

    使用 .stream() 方法
    """
    print("\n" + "=" * 70)
    print("示例 2：流式输出")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[calculator, get_weather],
        system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：北京天气如何？然后计算 10 加 20")
    print("\n流式输出（实时显示）：")
    print("-" * 70)

    # 使用 stream 方法
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    }, stream_mode="updates"):
        # 错误获取
        if 'messages' in chunk:
            # 获取最新的消息
            latest_msg = chunk['messages'][-1]

            # 如果是 AI 的最终回答
            if hasattr(latest_msg, 'content') and latest_msg.content:
                if not hasattr(latest_msg, 'tool_calls') or not latest_msg.tool_calls:
                    print(f"\n最终回答: {latest_msg.content}")

    print("\n关键点：")
    print("  - stream() 返回生成器，逐步返回结果")
    print("  - 用于实时显示进度")
    print("  - 适合长时间运行的任务")


def example_4_inspect_state():
    """
    示例4：在执行过程中查看状态

    使用 stream 并检查每个 chunk
    """
    print("\n" + "=" * 70)
    print("示例 4：查看中间状态")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：100 除以 5 等于多少？")
    print("\n执行步骤：")

    step = 0
    input = {"messages": [{"role": "user", "content": "100 除以 5 等于多少？"}]}
    for chunk in agent.stream(input, stream_mode="updates"):
        step += 1
        print(f"\n步骤 {step}:")
        latest = chunk
        print(latest)
        """
======================================================================
示例 4：查看中间状态
======================================================================

问题：100 除以 5 等于多少？

执行步骤：

步骤 1:
{'model': {'messages': [AIMessage(content='我来帮你计算100除以5等于多少。', additional_kwargs={}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'model_provider': 'deepseek'}, id='lc_run--019b6f65-4dd1-7d71-af4e-c3fb992556ac', tool_calls=[{'name': 'calculator', 'args': {'operation': 'divide', 'a': 100, 'b': 5}, 'id': 'call_00_WTqxkA9l70ZpvWhuSl0rQBkB', 'type': 'tool_call'}], usage_metadata={'input_tokens': 385, 'output_tokens': 82, 'total_tokens': 467, 'input_token_details': {'cache_read': 384}, 'output_token_details': {}})]}}

步骤 2:
{'tools': {'messages': [ToolMessage(content='100.0 divide 5.0 = 20.0', name='calculator', id='a201d603-1fdc-4e5a-84b3-0e98adcd465b', tool_call_id='call_00_WTqxkA9l70ZpvWhuSl0rQBkB')]}}

步骤 3:
{'model': {'messages': [AIMessage(content='100除以5等于20。', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'model_provider': 'deepseek'}, id='lc_run--019b6f65-5c5b-7fc1-be61-c2d4dac8e117', usage_metadata={'input_tokens': 497, 'output_tokens': 6, 'total_tokens': 503, 'input_token_details': {'cache_read': 448}, 'output_token_details': {}})]}}

        """
        # 错误获取，返回空
        # if 'messages' in chunk:
        #     latest = chunk['messages'][-1]
        #     msg_type = latest.__class__.__name__
        #     print(f"  类型: {msg_type}")
        #
        #     if hasattr(latest, 'tool_calls') and latest.tool_calls:
        #         print(f"  工具调用: {latest.tool_calls[0]['name']}")
        #     elif hasattr(latest, 'content') and latest.content:
        #         content_preview = latest.content[:50] if len(latest.content) > 50 else latest.content
        #         print(f"  内容: {content_preview}...")

    # print("\n关键点：")
    # print("  - stream 让你看到每个步骤")
    # print("  - 可以用于调试")
    # print("  - 可以用于进度显示")

def example_3_agent_with_system_prompt():
    def check_weather(location: str) -> str:
        '''Return the weather forecast for the specified location.'''
        return f"It's always sunny in {location}"


    graph = create_agent(
        model=model,
        tools=[check_weather],
        system_prompt="You are a helpful assistant",
    )
    inputs = {"messages": [{"role": "user", "content": "what is the weather in 上海，再介绍一下这个城市"}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        if "model" in chunk:
            print(type(chunk['model']['messages'][0].content))
            print(chunk['model']['messages'][0].content)




def main():
    try:
        # example_2_streaming()
        example_3_agent_with_system_prompt()
        # example_4_inspect_state()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
