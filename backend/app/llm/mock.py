"""A deterministic, rule-based stand-in for an LLM.

Purpose: keyless demo, CI, and exercising the *agent plumbing* (tool calls, memory, grounding, streaming)
end to end. It is NOT a measure of language-model quality: it understands a fixed set of phrasings via regexes.
Everything it says about jobs is read from tool results, so it can't hallucinate.
"""

import json
import re
from typing import Any

from app.agent.prompts import NOTICE_TAG, SUMMARY_SYSTEM
from app.ingest.skills import extract_skills
from app.llm.types import LLMResponse, Message, ToolSpec, Usage

_CITIES = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "gurugram": "Gurugram", "gurgaon": "Gurugram",
    "hyderabad": "Hyderabad", "pune": "Pune", "mumbai": "Mumbai", "noida": "Noida", "chennai": "Chennai",
    "delhi": "Delhi", "kolkata": "Kolkata", "ahmedabad": "Ahmedabad", "london": "London", "berlin": "Berlin",
    "singapore": "Singapore",
}
_ROLES = [
    (r"\bback[- ]?end\b", "backend engineer"), (r"\bfront[- ]?end\b", "frontend engineer"),
    (r"\bfull[- ]?stack\b", "full stack engineer"), (r"\bdata engineer", "data engineer"),
    (r"\bdata scien", "data scientist"), (r"\b(machine learning|ml engineer)\b", "machine learning engineer"),
    (r"\b(llm|ai engineer|genai)\b", "ai engineer"), (r"\bdevops|\bsre\b", "devops engineer"),
    (r"\bandroid\b", "android engineer"), (r"\bios\b", "ios engineer"), (r"\b(qa|sdet|tester)\b", "qa engineer"),
    (r"\bproduct manager|\bapm\b", "product manager"), (r"\bdesigner|\bux\b", "product designer"),
    (r"\banalyst\b", "data analyst"),
]
_SEARCH_WORDS = re.compile(r"\b(find|search|look(?:ing)? for|show|recommend|suggest|jobs?|roles?|openings?|internships?|positions?|opportunit\w+|hiring)\b")
_SELF = re.compile(r"\b(i know|i have|i've|i am (?:proficient|skilled|good)|i'm (?:proficient|skilled|good)|my skills|i can code|i work with|i use|experience (?:with|in)|comfortable with)\b")
_IDS = re.compile(r"#?\b(\d{1,9})\b")


def _user_text(m: Message) -> str:
    parts = [b["text"] for b in m.content if b["type"] == "text" and not b["text"].startswith(("<",))]
    return " ".join(parts).strip()


def _facts(text: str) -> dict[str, Any]:
    low = text.lower()
    f: dict[str, Any] = {}
    skills = extract_skills(text) if _SELF.search(low) else []
    if skills:
        f["skills"] = skills
    if "intern" in low and re.search(r"\b(i am|i'm|looking for|want|seeking|need|find)\b", low):
        f["job_type"], f["seniority"] = "internship", "intern"
    elif re.search(r"\b(student|fresher|final[- ]year|college|b\.?tech)\b", low):
        f["seniority"] = "junior"
    m = re.search(r"(\d+)\+?\s*years?", low)
    if m:
        y = int(m.group(1))
        f["seniority"] = "junior" if y <= 2 else "mid" if y <= 4 else "senior"
    cities = list(dict.fromkeys(v for k, v in _CITIES.items() if re.search(rf"\b{k}\b", low)))
    if cities:
        f["preferred_locations"] = cities
    if re.search(r"\b(remote(ly)?|work from home|wfh)\b", low):
        f["work_mode"] = "remote"
    elif re.search(r"\bhybrid\b", low):
        f["work_mode"] = "hybrid"
    elif re.search(r"\b(on-?site|in[- ]office)\b", low):
        f["work_mode"] = "onsite"
    roles = [r for pat, r in _ROLES if re.search(pat, low)]
    if roles:
        f["target_roles"] = roles[:3]
    m = re.search(r"(?:at least|minimum|min|above|more than|>)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?)", low)
    if m:
        f["min_salary"] = int(float(m.group(1)) * 100000)
    return f


def _tool(i: int, name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": f"toolu_mock_{i}", "name": name, "input": args}


class MockLLM:
    name = "mock"
    model = "mock-rule-based"

    # ------------------------------------------------------------ policy
    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 16000) -> LLMResponse:
        if system == SUMMARY_SYSTEM:
            body = messages[-1].content[0]["text"]
            return self._text("Conversation so far (condensed): " + " ".join(body.split())[:700])
        last = messages[-1]
        if any(b["type"] == "tool_result" for b in last.content):
            return self._compose(messages)
        text = _user_text(last)
        if any(b["type"] == "text" and b["text"].startswith(NOTICE_TAG) for b in last.content):
            return self._text("Sorry about that. Let me know what you'd like to look at and I'll search again.")
        return self._plan(text)

    def _plan(self, text: str) -> LLMResponse:
        low = text.lower()
        ids = [int(x) for x in _IDS.findall(low)]
        if re.search(r"\b(what|anything)\b.*\b(know|remember|saved)\b.*\b(about me|me)\b", low) and "jobs" not in low:
            return self._calls([("get_profile", {})])
        if re.search(r"\b(my saved|saved jobs|bookmarks)\b", low):
            return self._calls([("list_saved_jobs", {})])
        if ids and re.search(r"\bcompare\b", low) and len(ids) >= 2:
            return self._calls([("compare_jobs", {"job_ids": ids[:4]})])
        if ids and re.search(r"\b(save|bookmark)\b", low):
            return self._calls([("save_job", {"job_id": ids[0]})])
        if ids and re.search(r"\b(dismiss|hide|not interested)\b", low):
            return self._calls([("dismiss_job", {"job_id": ids[0]})])
        if ids and re.search(r"\b(why|explain|fit|match)\b", low):
            return self._calls([("explain_match", {"job_id": ids[0]})])
        if ids and re.search(r"\b(tell me (more )?about|details|describe|more about)\b", low):
            return self._calls([("get_job", {"job_id": ids[0]})])

        facts = _facts(text)
        calls: list[tuple[str, dict]] = []
        if facts:
            calls.append(("update_profile", facts))
        if _SEARCH_WORDS.search(low):
            # Like a real model would: search for the role asked about, not for the user's whole sentence
            # (which also lists skills they *have*, and would pull in unrelated jobs that merely mention them).
            query = " ".join(facts["target_roles"]) if facts.get("target_roles") else re.sub(r"\s+", " ", text)[:200]
            args: dict[str, Any] = {"query": query, "limit": 5}
            if "intern" in low:
                args["job_type"] = ["internship"]
            if facts.get("preferred_locations"):
                args["locations"], args["include_remote"] = facts["preferred_locations"], True
            if facts.get("work_mode") == "remote":
                args["work_mode"] = ["remote"]
            calls.append(("search_jobs", args))
        if calls:
            return self._calls(calls)
        return self._text("Tell me what you're looking for: your level, your skills, and a city or remote. "
                          "I'll find matching roles and remember your preferences.")

    def _calls(self, calls: list[tuple[str, dict]]) -> LLMResponse:
        blocks = [_tool(i, n, a) for i, (n, a) in enumerate(calls)]
        return LLMResponse(blocks, "tool_use", Usage(50, 20), self.model)

    def _text(self, text: str) -> LLMResponse:
        return LLMResponse([{"type": "text", "text": text}], "end_turn", Usage(50, 30), self.model)

    # ------------------------------------------------------------ composing answers from tool results
    def _compose(self, messages: list[Message]) -> LLMResponse:
        calls = {b["id"]: b for b in messages[-2].content if b["type"] == "tool_use"}
        parts: list[str] = []
        for r in (b for b in messages[-1].content if b["type"] == "tool_result"):
            call = calls.get(r["tool_use_id"], {"name": "?", "input": {}})
            if r.get("is_error"):
                parts.append(f"I hit a problem: {r['content'].removeprefix('Error: ')}")
                continue
            data = json.loads(r["content"])
            parts.append(getattr(self, f"_say_{call['name']}", self._say_default)(call["input"], data))
        return self._text("\n\n".join(p for p in parts if p))

    @staticmethod
    def _say_default(args, data) -> str:
        return "Done."

    @staticmethod
    def _say_update_profile(args, data) -> str:
        bits = [f"{k.replace('_', ' ')}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in args.items()]
        return "Noted, I'll remember: " + "; ".join(bits) + "."

    @staticmethod
    def _say_search_jobs(args, data) -> str:
        jobs = data["jobs"]
        if not jobs:
            return "I couldn't find matching jobs. Try loosening a filter (location, level) or different keywords."
        lines = [f"Here's what I found ({data['count']}):"]
        for i, j in enumerate(jobs[:5], 1):
            meta = ", ".join(x for x in (j["location"], j["seniority"], j["salary"]) if x)
            why = j["why"][0] if j["why"] else "matches your search"
            via = " (via Remotive)" if j.get("attribution") else ""
            warn = f" Watch out: {j['watch_out'][0]}." if j["watch_out"] else ""
            lines.append(f"{i}. #{j['id']} {j['title']} at {j['company']} ({meta}){via}. Why: {why}.{warn}")
        if not data["personalised"]:
            lines.append("Tell me your level, skills and preferred city and I'll rank these for you.")
        return "\n".join(lines)

    @staticmethod
    def _say_get_job(args, data) -> str:
        # Summarise structured fields only; never repeat free-text descriptions (untrusted third-party text).
        meta = ", ".join(x for x in (data["location"], data["seniority"], data["job_type"].replace("_", " "), data["salary"]) if x)
        return f"#{data['id']} {data['title']} at {data['company']} ({meta}). Skills: {', '.join(data['skills'])}. Apply: {data['apply_url']}"

    @staticmethod
    def _say_get_profile(args, data) -> str:
        p = data["profile"]
        known = [f"{k.replace('_', ' ')}: {', '.join(v) if isinstance(v, list) else v}" for k, v in p.items() if v not in (None, [], "")]
        return "Here's what I have saved about you: " + "; ".join(known) + "." if known else "I don't have anything saved about you yet."

    @staticmethod
    def _say_save_job(args, data) -> str:
        return f"Saved #{data['job_id']}."

    @staticmethod
    def _say_dismiss_job(args, data) -> str:
        return f"Dismissed #{data['job_id']}. I'll hide it and show fewer like it."

    @staticmethod
    def _say_list_saved_jobs(args, data) -> str:
        if not data["jobs"]:
            return "You haven't saved any jobs yet."
        return "Your saved jobs:\n" + "\n".join(f"- #{j['id']} {j['title']} at {j['company']}" for j in data["jobs"])

    @staticmethod
    def _say_compare_jobs(args, data) -> str:
        lines = ["Comparison:"]
        for j in data["jobs"]:
            extra = f"; you'd need: {', '.join(j['skills_to_learn'][:4])}" if j.get("skills_to_learn") else ""
            lines.append(f"- #{j['id']} {j['title']} at {j['company']} ({j['location']}, {j['seniority']}, {j['salary'] or 'salary n/a'}){extra}")
        if data["skills_in_all"]:
            lines.append(f"Common to all: {', '.join(data['skills_in_all'])}.")
        return "\n".join(lines)

    @staticmethod
    def _say_explain_match(args, data) -> str:
        s = f"Fit for #{data['job_id']} {data['title']}: {round(data['profile_fit'] * 100)}%."
        if data["strengths"]:
            s += " Strengths: " + "; ".join(data["strengths"]) + "."
        if data["concerns"]:
            s += " Concerns: " + "; ".join(data["concerns"]) + "."
        if data["profile_info_missing"]:
            s += " Tell me your " + ", ".join(data["profile_info_missing"]) + " for a sharper read."
        return s
