"""
piai LangChain integration.

Usage:
    from piai.langchain import PiAIChatModel

    llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")

    # Works anywhere LangChain accepts a chat model
    result = llm.invoke([HumanMessage(content="What is 2+2?")])
    async for chunk in llm.astream([HumanMessage(content="Tell me a joke")]):
        print(chunk.content, end="", flush=True)

Requires:
    pip install langchain-core
"""

from .chat_model import PiAIChatModel
from .sub_agent_tool import SubAgentTool

__all__ = ["PiAIChatModel", "SubAgentTool"]
