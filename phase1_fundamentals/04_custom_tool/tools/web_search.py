# !/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
# ----------------------------------------------------------------------------------------------------------------------

from langchain_core.tools import tool
from typing import Optional
import json


@tool
def web_search(query: str, count_results: Optional[int] = 2) -> str:

    """
    模拟在网上搜索消息
    :param query:输入的关键字
    :param count_results:可选参数，默认是2
    :return:字符串形式的结果
    """

    mock_results = {
        "Python": [
            "Python官方网站 - https://www.python.org",
            "Python教程 - 菜鸟教程",
            "Python最佳实践 - Real Python"
        ],
        "机器学习": [
            "机器学习入门 - Coursera",
            "Scikit-learn文档",
            "机器学习实战 - GitHub"
        ],
        "LangChain": [
            "LangChain官方文档",
            "LangChain GitHub仓库",
            "LangChain教程 - YouTube"
        ]
    }

    # 查找结果
    results = []

    for key in mock_results:
        if key.lower() in query.lower():
            results = mock_results[key][:count_results]
            break

    if not results:
        return f"未找到关于'{query}'的结果"

    # 格式化输出
    output = f"搜索 '{query}' 找到 {len(results)} 条结果：\n"

    for i, results in enumerate(results, 1):
        output += f"[{i}] {results}\n"

    return json.dumps(output,ensure_ascii=False).strip()


# 测试工具
if __name__ == "__main__":
    print("测试搜索工具：")
    print(web_search.invoke({"query": "Python"}))
    print("\n" + web_search.invoke({"query": "LangChain", "count_results": 1}))

