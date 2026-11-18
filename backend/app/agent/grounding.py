"""Hallucination guard: every job the assistant cites by id (#123) must have come back from a tool this turn."""

import re
from dataclasses import dataclass

_REF = re.compile(r"(?<![\w&/])#(\d{1,9})\b")


@dataclass
class Grounding:
    cited: list[int]
    unknown: list[int]  # cited but never returned by any tool call this turn

    @property
    def ok(self) -> bool:
        return not self.unknown


def check_grounding(text: str, seen_job_ids: set[int]) -> Grounding:
    cited = list(dict.fromkeys(int(m) for m in _REF.findall(text)))
    return Grounding(cited, [c for c in cited if c not in seen_job_ids])


def annotate_unverified(text: str, unknown: list[int]) -> str:
    """Last resort when the model keeps citing ids it never retrieved: don't present them as fact."""
    for jid in unknown:
        text = re.sub(rf"(?<![\w&/])#{jid}\b", f"#{jid} (unverified)", text)
    return text
