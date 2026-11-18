"""The retrieval eval set: queries + a graded relevance judgment for EVERY job in the corpus.

Judgments are functions of the job's ground-truth metadata (role family, skill list, level, location ...),
never of the text the search engine reads. Grades: 0 irrelevant, 1 related, 3 exactly on target
(personalised queries use a 0..6 scale; see `personalised`).

Categories
  role_title    plain job titles                         paraphrase   describes the job without naming it
  skills        skill lists                              alias        abbreviations / jargon
  constrained   query + hard filters (SQL)               personalised query + saved profile (re-ranker)

Split: ~1/3 of queries are "test"; any tuning was done on "dev" only (see EVALS.md).

Honest caveat: the corpus is synthetic and generated from the same taxonomy these judgments use, so absolute
numbers are an upper bound and a regression harness, not a claim about real-world performance.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from app.ingest.skills import extract_skills

Record = dict
Judge = Callable[[Record], float]

RELATED = {
    "backend": {"fullstack"}, "frontend": {"fullstack"}, "fullstack": {"backend", "frontend"},
    "ml": {"llm", "data_eng"}, "llm": {"ml"}, "data_eng": {"ml", "analyst"}, "analyst": {"data_eng", "ml"},
    "devops": {"security"}, "security": {"devops"},
}


def family(primary: set[str], related: set[str] | None = None) -> Judge:
    def judge(r: Record) -> float:
        if r["role_family"] in primary:
            return 3
        return 1 if related and r["role_family"] in related else 0
    return judge


def skills(wanted: list[str]) -> Judge:
    def judge(r: Record) -> float:
        n = len(set(r["skills"]) & set(wanted))
        return 0 if n == 0 else 1 if n == 1 else 3
    return judge


def only_if(base: Judge, predicate: Callable[[Record], bool]) -> Judge:
    return lambda r: base(r) if predicate(r) else 0


def personalised(base: Judge, profile: dict) -> Judge:
    """Topical relevance x how well the job fits the person: level, job type, location, skills (0..6)."""
    prefs = {p.lower() for p in profile.get("preferred_locations", [])}

    def judge(r: Record) -> float:
        b = base(r)
        if b == 0:
            return 0
        fit = 0
        fit += profile.get("seniority") == r["seniority"]
        fit += profile.get("job_type") is not None and profile["job_type"] == r["job_type"]
        fit += bool(prefs) and (r["location"].lower() in prefs or r["work_mode"] == "remote" and "remote" in prefs)
        fit += len(set(profile.get("skills", [])) & set(r["skills"])) >= 2
        if profile.get("work_mode"):
            fit += profile["work_mode"] == r["work_mode"]
        return b * (0.5 + fit / 2)  # no fit -> half credit; full fit on 4 factors -> 2.5x
    return judge


@dataclass
class EvalQuery:
    id: str
    text: str
    category: str
    judge: Judge
    filters: dict = field(default_factory=dict)
    profile: dict | None = None  # present => personalised query
    topical: Judge | None = None  # personalised queries: the profile-blind judgment, to check relevance isn't sacrificed
    split: str = "dev"


def _q(id_, text, category, judge, **kw) -> EvalQuery:
    return EvalQuery(id_, text, category, judge, **kw)


def build_queries() -> list[EvalQuery]:
    Q: list[EvalQuery] = []
    add = lambda *a, **k: Q.append(_q(*a, **k))  # noqa: E731

    # --- role titles
    add("role-backend", "backend engineer", "role_title", family({"backend"}, RELATED["backend"]))
    add("role-frontend", "frontend developer", "role_title", family({"frontend"}, RELATED["frontend"]))
    add("role-fullstack", "full stack developer", "role_title", family({"fullstack"}, RELATED["fullstack"]))
    add("role-ml", "machine learning engineer", "role_title", family({"ml"}, RELATED["ml"]))
    add("role-dataeng", "data engineer", "role_title", family({"data_eng"}, RELATED["data_eng"]))
    add("role-devops", "devops engineer", "role_title", family({"devops"}, RELATED["devops"]))
    add("role-android", "android developer", "role_title", family({"mobile"}))
    add("role-qa", "qa automation engineer", "role_title", family({"qa"}))
    add("role-pm", "product manager", "role_title", family({"pm"}))
    add("role-ux", "ux designer", "role_title", family({"design"}))
    add("role-analyst", "data analyst", "role_title", family({"analyst"}, RELATED["analyst"]))
    add("role-security", "security engineer", "role_title", family({"security"}, RELATED["security"]))
    add("role-ai", "AI engineer", "role_title", family({"llm"}, RELATED["llm"]))

    # --- skill lists (judged by skill overlap, independent of role family)
    for id_, text in [
        ("skill-k8s", "kubernetes terraform helm"), ("skill-react", "react typescript redux"),
        ("skill-dl", "pytorch tensorflow scikit-learn"), ("skill-spark", "spark airflow dbt"),
        ("skill-test", "selenium pytest cypress"), ("skill-figma", "figma prototyping design systems"),
        ("skill-kotlin", "kotlin jetpack compose"), ("skill-bi", "tableau power bi excel"),
        ("skill-sec", "owasp penetration testing threat modeling"),
        ("skill-rag", "rag embeddings vector database langchain"), ("skill-pyapi", "python fastapi postgresql"),
    ]:
        add(id_, text, "skills", skills(extract_skills(text)))  # same canonicalisation the engine uses

    # --- paraphrases: describe the job without using its usual title or keywords
    add("para-llm", "build chatbots with large language models", "paraphrase", family({"llm"}, RELATED["llm"]))
    add("para-mobile", "apps for iphone and android phones", "paraphrase", family({"mobile"}))
    add("para-sre", "keep servers running reliably", "paraphrase", family({"devops"}, RELATED["devops"]))
    add("para-analyst", "turn data into business insights", "paraphrase", family({"analyst"}, RELATED["analyst"]))
    add("para-design", "design user interfaces and run usability tests", "paraphrase", family({"design"}))
    add("para-ml", "create models that make predictions from data", "paraphrase", family({"ml"}, RELATED["ml"]))
    add("para-sec", "protect applications from hackers", "paraphrase", family({"security"}, RELATED["security"]))
    add("para-pm", "plan the roadmap and talk to customers", "paraphrase", family({"pm"}))

    # --- aliases / abbreviations
    add("alias-k8s", "k8s", "alias", skills(["kubernetes"]))
    add("alias-ml", "ML engineer", "alias", family({"ml"}, RELATED["ml"]))
    add("alias-sre", "SRE", "alias", family({"devops"}, RELATED["devops"]))
    add("alias-genai", "genai engineer", "alias", family({"llm"}, RELATED["llm"]))
    add("alias-sde-intern", "sde intern", "alias", lambda r: 3 if r["seniority"] == "intern" else 0)
    add("alias-fe-intern", "frontend internship", "alias",
        lambda r: 3 if r["role_family"] == "frontend" and r["seniority"] == "intern"
        else 1 if r["role_family"] == "frontend" or r["seniority"] == "intern" else 0)

    # --- constrained: hard filters must hold exactly; relevance is topical AND constraint-satisfying
    add("con-backend-pune", "backend engineer", "constrained",
        only_if(family({"backend"}, RELATED["backend"]), lambda r: r["location"] == "Pune"), filters={"locations": ["Pune"]})
    add("con-ds-remote", "data scientist", "constrained",
        only_if(family({"ml"}, {"llm", "analyst"}), lambda r: r["work_mode"] == "remote"), filters={"work_mode": ["remote"]})
    add("con-intern-gurugram", "internship", "constrained",
        lambda r: 3 if r["job_type"] == "internship" and r["location"] == "Gurugram" else 0,
        filters={"job_type": ["internship"], "locations": ["Gurugram"]})
    add("con-pm-junior-mid", "product manager", "constrained",
        only_if(family({"pm"}), lambda r: r["seniority"] in ("junior", "mid")), filters={"seniority": ["junior", "mid"]})
    add("con-python-fresh-remote", "python developer", "constrained",
        only_if(skills(["python"]), lambda r: r["work_mode"] == "remote" and r["posted_days_ago"] <= 14),
        filters={"work_mode": ["remote"], "posted_within_days": 14})

    # --- personalised: query + saved profile; judged on topical relevance x personal fit
    def pers(id_, text, base, profile):
        add(id_, text, "personalised", personalised(base, profile), profile=profile, topical=base)

    pers("pers-intern-backend", "backend jobs", family({"backend"}, RELATED["backend"]),
         {"skills": ["python", "sql"], "seniority": "intern", "job_type": "internship", "preferred_locations": ["Gurugram", "Bengaluru"]})
    pers("pers-senior-swe", "software engineer", family({"backend", "devops"}, {"fullstack"}),
         {"skills": ["java", "kubernetes", "aws"], "seniority": "senior", "work_mode": "remote"})
    pers("pers-junior-ml", "data jobs", family({"ml", "data_eng", "analyst"}),
         {"skills": ["python", "pytorch"], "seniority": "junior", "preferred_locations": ["Bengaluru"]})
    pers("pers-mid-design", "design jobs", family({"design"}),
         {"skills": ["figma"], "seniority": "mid", "work_mode": "remote"})
    pers("pers-frontend-reco", "", family({"frontend"}, RELATED["frontend"]),
         {"target_roles": ["frontend engineer"], "skills": ["react", "css", "javascript"], "seniority": "junior", "preferred_locations": ["Pune"]})
    pers("pers-mobile-intern", "engineering roles", family({"mobile"}),
         {"skills": ["kotlin", "android sdk"], "seniority": "intern", "job_type": "internship", "preferred_locations": ["Hyderabad"]})

    for i, q in enumerate(Q):
        q.split = "test" if i % 3 == 2 else "dev"

    # Appended LAST on purpose (adding earlier would shift the dev/test assignment of every later query). This is the
    # exact query that visibly failed in a demo, added after the failure was seen: a sanity canary, not independent
    # evidence. Its split is "canary": excluded from every aggregate and from all tuning, reported on its own.
    canary = _q("con-frontend-remote", "frontend engineer", "constrained",
                only_if(family({"frontend"}, RELATED["frontend"]), lambda r: r["work_mode"] == "remote"),
                filters={"work_mode": ["remote"]})
    canary.split = "canary"
    Q.append(canary)
    return Q
