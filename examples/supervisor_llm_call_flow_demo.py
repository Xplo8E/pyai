"""
Example: tiny supervisor flow showing real LLM/tool call behavior end-to-end.

Goal:
- Mirror fuzzer-style flow where a supervisor calls a `report_findings` tool.
- Inside that tool, use `PiAIChatModel.with_structured_output()`.
- Print actual message sequence so you can see how calls happen.

Run:
  cd /Users/vinay/pyai
  PYTHONPATH=/Users/vinay/pyai/src .venv/bin/python examples/supervisor_llm_call_flow_demo.py
"""

from __future__ import annotations

import asyncio
import json

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel

from piai.langchain import PiAIChatModel


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class FindingReport(BaseModel):
    library_name: str
    targets_found: int
    verdict: str
    summary: str


async def main() -> None:
    report_tool_invocations = {"count": 0}

    @tool
    def report_findings(analysis_note: str) -> str:
        """Generate a structured finding report from analysis notes."""
        report_tool_invocations["count"] += 1
        print(f"[tool] report_findings called (count={report_tool_invocations['count']})")
        print(f"[tool] analysis_note: {analysis_note}")

        llm = PiAIChatModel(model_name="gpt-5.1-codex-mini")
        chain = llm.with_structured_output(FindingReport, include_raw=True)

        out = chain.invoke(
            [
                HumanMessage(
                    content=(
                        "Convert this analysis note into FindingReport JSON. "
                        f"Note: {analysis_note}"
                    )
                )
            ]
        )

        parsed: FindingReport = out["parsed"]
        print("[tool] parsed FindingReport:", parsed.model_dump())
        print("[tool] raw message type:", type(out["raw"]).__name__)
        return json.dumps(parsed.model_dump())

    worker = create_agent(
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        tools=[],
        system_prompt="You are a helper agent.",
        name="worker",
    )

    workflow = create_supervisor(
        agents=[worker],
        tools=[report_findings],
        model=PiAIChatModel(model_name="gpt-5.1-codex-mini"),
        prompt=(
            "You are a fuzzing supervisor. "
            "Always call report_findings once using the user's analysis context, "
            "then provide a short completion message. "
            "Do not transfer to worker unless report_findings is impossible."
        ),
        output_mode="last_message",
    )

    app = workflow.compile()

    result = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Analyze libcrypto.so quickly. "
                        "Potential overflow near JNI parser path. "
                        "Call report_findings and complete."
                    )
                )
            ]
        }
    )

    final_msg = result["messages"][-1]

    print("\n=== Final Response ===")
    print(final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content))
    print()

    print("=== Message Sequence ===")
    for i, msg in enumerate(result["messages"], start=1):
        print(f"{i}. {type(msg).__name__}")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print(f"   tool_calls: {tool_calls}")
        content = getattr(msg, "content", None)
        if content:
            preview = content if isinstance(content, str) else str(content)
            print(f"   content: {preview[:240]}")

    _require(report_tool_invocations["count"] >= 1, "report_findings tool was never called")
    print(f"\n[OK] report_findings invocation count = {report_tool_invocations['count']}")
    print("[OK] Supervisor + structured-output real flow completed.")


if __name__ == "__main__":
    asyncio.run(main())
