"""
Example: PiAIChatModel.with_structured_output() in a fuzzer-style flow.

This demo is REAL execution (no mocking).

Run:
  PYTHONPATH=src .venv/bin/python examples/structured_output_fuzzer_style.py
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from piai.langchain.chat_model import PiAIChatModel
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class FindingReport(BaseModel):
    library_name: str
    targets_found: int
    verdict: str


def demo_parsed_only() -> None:
    model = PiAIChatModel(model_name="gpt-5.1-codex-mini")
    chain = model.with_structured_output(FindingReport)
    result = chain.invoke([
        HumanMessage(
            content=(
                "Return a FindingReport JSON for a native analysis result. "
                "Use: library_name=libcrypto.so, targets_found=3, verdict=candidate, "
                "summary='JNI boundary + memcpy on user-controlled data'."
            )
        )
    ])
    _require(isinstance(result, FindingReport), "Expected FindingReport instance")
    _require(result.library_name != "", "library_name should not be empty")
    _require(result.verdict != "", "verdict should not be empty")

    print("[OK] parsed-only mode")
    print("     report:", result.model_dump())


def demo_include_raw() -> None:
    model = PiAIChatModel(model_name="gpt-5.1-codex-mini")
    chain = model.with_structured_output(FindingReport, include_raw=True)
    result = chain.invoke([
        HumanMessage(
            content=(
                "Return a FindingReport JSON for: library_name=libssl.so, "
                "targets_found=1, verdict=confirmed, "
                "summary='bounded memcpy but reachable with attacker input'."
            )
        )
    ])
    _require(isinstance(result, dict), "Expected dict output when include_raw=True")
    _require("raw" in result and "parsed" in result, "Expected {'raw','parsed'} keys")
    _require(isinstance(result["parsed"], FindingReport), "parsed must be FindingReport")

    print("[OK] include_raw mode")
    print("     parsed:", result["parsed"].model_dump())
    print("     raw type:", type(result["raw"]).__name__)


def demo_typed_dict_schema() -> None:
    class TargetSummary(TypedDict):
        library_name: str
        targets_found: int

    model = PiAIChatModel(model_name="gpt-5.1-codex-mini")
    chain = model.with_structured_output(TargetSummary)
    result = chain.invoke([
        HumanMessage(
            content=(
                "Return TargetSummary JSON with library_name and targets_found only. "
                "Use values: library_name=libfoo.so, targets_found=2."
            )
        )
    ])
    _require(isinstance(result, dict), "TypedDict schema should return dict")
    _require("library_name" in result and "targets_found" in result, "Missing expected keys")

    print("[OK] typed-dict schema mode")
    print("     parsed:", result)


def demo_openai_tool_dict_schema() -> None:
    schema = {
        "name": "FindingSchema",
        "description": "Structured finding summary",
        "parameters": {
            "type": "object",
            "properties": {
                "library": {"type": "string"},
                "verdict": {"type": "string"},
            },
            "required": ["library", "verdict"],
        },
    }

    model = PiAIChatModel(model_name="gpt-5.1-codex-mini")
    chain = model.with_structured_output(schema)
    result = chain.invoke([
        HumanMessage(
            content=(
                "Return FindingSchema JSON with fields library and verdict only. "
                "Use values: library=libbar.so, verdict=candidate."
            )
        )
    ])
    _require(isinstance(result, dict), "dict schema should return dict")
    _require("library" in result and "verdict" in result, "Missing expected keys")

    print("[OK] openai-tool-dict schema mode")
    print("     parsed:", result)


def main() -> None:
    demo_parsed_only()
    demo_include_raw()
    demo_typed_dict_schema()
    demo_openai_tool_dict_schema()
    print("\nExample completed: structured output behavior matches expected Phase-0 contract.")


if __name__ == "__main__":
    main()
