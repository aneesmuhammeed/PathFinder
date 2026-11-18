"""Text normalisation helpers for messy third-party job data."""

import re
from html import unescape
from html.parser import HTMLParser

_BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}
_SKIP = {"script", "style"}


class _Stripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Strip tags, keep paragraph/list structure, collapse whitespace."""
    if "<" not in html:
        text = unescape(html)
    else:
        p = _Stripper()
        p.feed(html)
        p.close()
        text = "".join(p.parts)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- salary ----------

_CURRENCY = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "₹": "INR", "inr": "INR"}
_SALARY = re.compile(
    r"(?P<cur1>[$€£₹]|usd|eur|gbp|inr)?\s*(?P<lo>\d[\d,\.]*)\s*(?P<lok>k)?\s*(?:-|–|to)\s*"
    r"(?P<cur2>[$€£₹]|usd|eur|gbp|inr)?\s*(?P<hi>\d[\d,\.]*)\s*(?P<hik>k)?\s*(?P<cur3>usd|eur|gbp|inr)?",
    re.I,
)


def _num(s: str, k: str | None) -> float | None:
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    return v * 1000 if k else v


def parse_salary(text: str | None) -> tuple[int, int, str, str] | None:
    """Best-effort '$80k - $120k' → (80000, 120000, 'USD', 'year'). None when unsure."""
    if not text:
        return None
    low = text.lower()
    if "hour" in low or "/hr" in low or "per day" in low:
        return None
    m = _SALARY.search(text)
    if not m:
        return None
    cur_raw = (m["cur1"] or m["cur2"] or m["cur3"] or "").lower()
    cur = _CURRENCY.get(cur_raw)
    lo, hi = _num(m["lo"], m["lok"] or m["hik"]), _num(m["hi"], m["hik"])
    if cur is None or lo is None or hi is None or hi < lo:
        return None
    period = "month" if ("month" in low or "/mo" in low) else "year"
    floor = 1000 if period == "month" else 10000  # ignore things like "5-10" (years of experience)
    if lo < floor:
        return None
    return int(lo), int(hi), cur, period


# ---------- classification ----------

_RE = lambda *words: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)  # noqa: E731
_INTERN = _RE("intern", "internship", "trainee", "werkstudent\\w*", "working student", "apprentice")
_LEAD = _RE("lead", "head", "director", "manager", "principal", "architect", "vp")
_SENIOR = _RE("senior", "sr\\.?", "staff", "expert")
_JUNIOR = _RE("junior", "jr\\.?", "entry[- ]level", "graduate", "associate", "fresher")


def infer_seniority(title: str, hints: str = "") -> str:
    t = title
    if _INTERN.search(t) or _INTERN.search(hints):
        return "intern"
    if re.search(r"\bproduct manager\b", t, re.I) and not _SENIOR.search(t) and not re.search(r"\b(group|director|head)\b", t, re.I):
        return "mid"  # "Product Manager" is a role, not a management level
    if re.search(r"\bengineering manager\b|\b(head|director|vp)\b|\bprincipal\b|\blead\b|\barchitect\b", t, re.I):
        return "lead"
    if _SENIOR.search(t):
        return "senior"
    if _JUNIOR.search(t) or re.search(r"\bentry\b|\bberufseinsteiger", hints, re.I):
        return "junior"
    return "mid"


def infer_job_type(*texts: str) -> str:
    blob = " ".join(texts).lower()
    if _INTERN.search(blob):
        return "internship"
    if re.search(r"\b(contract|freelance|befristet\w*|temporary)\b", blob):
        return "contract"
    if re.search(r"\bpart[- ]?time\b|\bteilzeit\b", blob):
        return "part_time"
    return "full_time"


_TECH_TITLE = _RE(
    "engineer\\w*", "developer\\w*", "software", "data", "devops", "sre", "security", "scientist", "analyst",
    "architect", "programmer", "qa", "sdet", "machine learning", "ai", "ml", "llm", "frontend", "front[- ]end",
    "backend", "back[- ]end", "full[- ]?stack", "web", "mobile", "android", "ios", "cloud", "platform",
    "product manager", "product owner", "designer", "ux", "ui", "python", "java", "react", "it",
)


_NON_TECH_TITLE = re.compile(
    r"\b(sales|marketing|go[- ]to[- ]market|gtm|recruit\w*|account (?:executive|manager)|customer (?:success|support|service)|"
    r"nurse|pflege\w*|writer|copywriter|accountant|bookkeep\w*|legal|paralegal|driver|warehouse|cook|chef)\b",
    re.I,
)


def is_tech_role(title: str, skills: list[str]) -> bool:
    """Heuristic filter for the engineering/data/product/design roles this product serves."""
    if _NON_TECH_TITLE.search(title):
        return False
    return bool(_TECH_TITLE.search(title)) or len(skills) >= 3


_GERMAN = frozenset("und der die das wir mit für sind eine einen ihre ihr deine du dich bei oder auf zu von im ist werden".split())
_ENGLISH = frozenset("and the we you with for are our your will to of in is on as a an this that have".split())


def looks_english(text: str) -> bool:
    """Cheap language guess (stop-word ratio); good enough to drop German postings from Arbeitnow."""
    words = re.findall(r"[a-zäöüß]+", text.lower()[:3000])
    if len(words) < 20:
        return True
    de = sum(w in _GERMAN for w in words)
    en = sum(w in _ENGLISH for w in words)
    return en >= de


# ---------- location ----------

_COUNTRIES = {
    "India": ["india", "bengaluru", "bangalore", "gurugram", "gurgaon", "hyderabad", "pune", "mumbai", "noida",
              "chennai", "delhi", "kolkata", "ahmedabad"],
    "Germany": ["germany", "deutschland", "berlin", "munich", "münchen", "hamburg", "köln", "cologne", "frankfurt",
                "stuttgart", "düsseldorf", "leipzig", "dresden", "kiel"],
    "United Kingdom": ["united kingdom", "uk", "london", "manchester", "england"],
    "United States": ["united states", "usa", "us", "new york", "san francisco", "california", "texas", "austin"],
    "Singapore": ["singapore"],
    "Canada": ["canada", "toronto", "vancouver"],
}


def infer_country(location: str | None) -> str | None:
    if not location:
        return None
    low = location.lower()
    for country, needles in _COUNTRIES.items():
        for n in needles:
            if re.search(rf"\b{re.escape(n)}\b", low):
                return country
    return None
