"""Groundedness checks: a free deterministic one, and an optional LLM judge.

Deterministic (always on): every salary figure the assistant states must be traceable to a tool result (or to
something the user said). Heuristic: it flags rounding ("around ₹20 LPA") as a false positive, so treat it as a
tripwire for invented numbers, not a proof of correctness.

LLM judge (opt-in, costs money): a second model call that checks each answer against the tool outputs.
Caveat: judging with the same model family that wrote the answer is biased toward agreeing with itself.
"""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.llm.types import LLMClient, LLMError, Message

_NUM = r"\d[\d,]*(?:\.\d+)?"
_UNIT = r"(?:LPA|lpa|/month|/year|per month|per year)"
_MONEY = re.compile(
    rf"(?:₹|\$|USD|EUR|GBP|INR)\s?{_NUM}(?:\s?[–-]\s?{_NUM})?(?:\s?{_UNIT})?|{_NUM}(?:\s?[–-]\s?{_NUM})?\s?{_UNIT}"
)
_NUMBER = re.compile(_NUM)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def ungrounded_numbers(answer: str, tool_outputs: list[str], user_texts: list[str] | None = None) -> list[str]:
    """Salary-like numbers in `answer` that appear in no tool output and in nothing the user said."""
    allowed = _numbers(" ".join(tool_outputs)) | _numbers(" ".join(user_texts or []))
    bad = []
    for m in _MONEY.finditer(answer):
        for n in _numbers(m.group(0)):
            if n not in allowed:
                bad.append(m.group(0).strip())
                break
    return bad


JUDGE_SYSTEM = """You are a strict fact-checker for a job-search assistant.
You get the user's request, the raw outputs of the tools the assistant called, and the assistant's final answer.
Decide whether every factual claim about specific jobs (titles, companies, locations, salaries, requirements, ids, links) in the answer is supported by the tool outputs. Opinions, advice and questions need no support.
Reply with ONLY a JSON object: {"supported": true|false, "unsupported_claims": ["..."]}"""


class Verdict(BaseModel):
    supported: bool
    unsupported_claims: list[str] = []


@dataclass
class JudgeResult:
    verdict: Verdict | None  # None = the judge failed or returned something unparsable (reported as "unscored")
    raw: str = ""


def parse_verdict(text: str) -> Verdict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return Verdict.model_validate(json.loads(match.group(0)))
    except (ValueError, ValidationError):
        return None


def judge_answer(llm: LLMClient, question: str, tool_outputs: list[str], answer: str) -> JudgeResult:
    body = (f"<user_request>\n{question}\n</user_request>\n\n<tool_outputs>\n" + "\n---\n".join(tool_outputs)[:14000]
            + f"\n</tool_outputs>\n\n<assistant_answer>\n{answer}\n</assistant_answer>")
    try:
        resp = llm.complete(system=JUDGE_SYSTEM, messages=[Message("user", [{"type": "text", "text": body}])], tools=[], max_tokens=2000)
    except LLMError as exc:
        return JudgeResult(None, f"judge error: {exc}")
    return JudgeResult(parse_verdict(resp.text), resp.text)
