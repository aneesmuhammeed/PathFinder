"""Generate data/seed_jobs.jsonl: a deterministic, realistic synthetic jobs corpus (~500 postings).

    python scripts/generate_seed.py

Why synthetic? It lets the project run fully offline, gives the eval harness a corpus whose
ground truth we control (every job has an explicit role family + skill list), and the live
ingestion adapters (Remotive / Arbeitnow) provide real, constantly-changing data on top.
All companies are fictional. Output is deterministic (fixed RNG seed), so evals are reproducible.
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "data" / "seed_jobs.jsonl"
N_JOBS = 520

COMPANIES = [
    "Zenvoy", "Kiteflow", "Brightloom", "Quantaro", "Nimbus Labs", "Paperplane", "Orbitly", "Hexaware Systems",
    "Lumora", "Cartwheel", "Fintrail", "Medilink", "Stackwise", "Vantora", "Pixelforge", "Greenleaf Tech",
    "Datavine", "Cloudharbor", "Rupeeo", "Swiftcart", "Learnly", "Tripnest", "Mapleware", "Ironbridge Digital",
    "Sparrow AI", "Beacon Health", "Novapay", "Terrabyte", "Blueprint Robotics", "Alchemy Games", "Skyroute",
    "Inkwell Media", "Kernel Works", "Ledgerly", "Foodloop", "Cobalt Security", "Harbor Analytics", "Zipline Labs",
    "Aurora Commerce", "Tessellate", "Gridiron Energy", "Mosaic Learning", "Pathwise", "Redwood Cloud",
    "Silverline Pay", "Trellis HR", "Vividly", "Wavelength", "Yonder Travel", "Bytecraft",
]

# (city, country, weight) - heavy on India since the target product serves Indian early-career users.
LOCATIONS = [
    ("Bengaluru", "India", 22), ("Gurugram", "India", 12), ("Hyderabad", "India", 9), ("Pune", "India", 8),
    ("Mumbai", "India", 6), ("Noida", "India", 5), ("Chennai", "India", 4), ("Delhi", "India", 4),
    ("Kolkata", "India", 2), ("Ahmedabad", "India", 2), ("Remote (India)", "India", 8),
    ("London", "United Kingdom", 3), ("Berlin", "Germany", 3), ("Singapore", "Singapore", 3),
    ("San Francisco", "United States", 3), ("New York", "United States", 2), ("Remote (Global)", None, 4),
]

# Each family: titles per seniority, core skills (always), optional skills (sampled), responsibilities, domains.
FAMILIES = {
    "backend": {
        "titles": {"intern": ["Backend Engineering Intern", "SDE Intern - Backend"],
                   "junior": ["Backend Engineer", "Software Engineer - Backend"],
                   "mid": ["Backend Engineer II", "Software Engineer (Backend)"],
                   "senior": ["Senior Backend Engineer", "Staff Software Engineer, Platform"],
                   "lead": ["Backend Engineering Lead", "Engineering Manager, Backend"]},
        "core": ["python", "sql", "rest apis"],
        "optional": ["fastapi", "django", "postgresql", "redis", "docker", "kubernetes", "aws", "kafka",
                     "microservices", "celery", "graphql", "go", "java", "node.js", "mongodb", "git"],
        "resp": [
            "Design, build and operate backend services that serve millions of requests a day",
            "Model data in relational databases and write efficient, well-indexed queries",
            "Own the reliability of your services: monitoring, alerting, incident follow-ups",
            "Design clean REST APIs that mobile and web teams enjoy using",
            "Profile and optimise slow endpoints and database queries",
            "Write automated tests and participate in thorough code reviews",
        ],
    },
    "frontend": {
        "titles": {"intern": ["Frontend Engineering Intern", "SDE Intern - Web"],
                   "junior": ["Frontend Engineer", "Software Engineer - Web"],
                   "mid": ["Frontend Engineer II", "UI Engineer"],
                   "senior": ["Senior Frontend Engineer", "Staff Frontend Engineer"],
                   "lead": ["Frontend Lead", "Engineering Manager, Web"]},
        "core": ["javascript", "react", "html", "css"],
        "optional": ["typescript", "next.js", "redux", "tailwind css", "webpack", "jest", "graphql", "figma",
                     "accessibility", "git", "vue.js"],
        "resp": [
            "Build fast, accessible user interfaces in React with a shared design system",
            "Turn Figma designs into pixel-perfect, responsive components",
            "Improve Core Web Vitals and bundle size across our web app",
            "Write unit and integration tests for UI components",
            "Work with designers and backend engineers to ship features end to end",
        ],
    },
    "fullstack": {
        "titles": {"intern": ["Full Stack Intern", "SDE Intern - Full Stack"],
                   "junior": ["Full Stack Engineer", "Software Development Engineer I"],
                   "mid": ["Full Stack Engineer II", "Software Development Engineer II"],
                   "senior": ["Senior Full Stack Engineer", "Senior Software Engineer"],
                   "lead": ["Tech Lead, Full Stack", "Principal Engineer"]},
        "core": ["javascript", "sql", "rest apis"],
        "optional": ["react", "node.js", "typescript", "python", "postgresql", "mongodb", "docker", "aws",
                     "next.js", "express", "git", "redis"],
        "resp": [
            "Ship product features across the stack, from database schema to React UI",
            "Build APIs and the interfaces that consume them",
            "Take features from idea to production and measure their impact",
            "Keep deployments fast and safe with CI/CD and feature flags",
            "Pair with product managers to scope work and cut unnecessary complexity",
        ],
    },
    "ml": {
        "titles": {"intern": ["Machine Learning Intern", "Data Science Intern"],
                   "junior": ["Machine Learning Engineer", "Associate Data Scientist"],
                   "mid": ["Machine Learning Engineer II", "Data Scientist"],
                   "senior": ["Senior Machine Learning Engineer", "Senior Data Scientist"],
                   "lead": ["Lead ML Engineer", "Head of Data Science"]},
        "core": ["python", "machine learning", "sql"],
        "optional": ["pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "mlops", "feature engineering",
                     "a/b testing", "spark", "docker", "airflow", "xgboost", "statistics"],
        "resp": [
            "Train, evaluate and ship models that power ranking, recommendations and forecasting",
            "Run offline experiments and online A/B tests to measure real user impact",
            "Build feature pipelines and monitor models for drift in production",
            "Translate fuzzy business problems into well-posed ML problems",
            "Write reproducible, well-tested training and evaluation code",
        ],
    },
    "llm": {
        "titles": {"intern": ["AI Engineering Intern", "LLM Applications Intern"],
                   "junior": ["AI Engineer", "Applied AI Engineer"],
                   "mid": ["LLM Engineer", "Applied AI Engineer II"],
                   "senior": ["Senior AI Engineer", "Staff Engineer, LLM Platform"],
                   "lead": ["AI Engineering Lead", "Head of Applied AI"]},
        "core": ["python", "llm", "prompt engineering"],
        "optional": ["rag", "embeddings", "vector databases", "langchain", "fastapi", "pytorch", "evals",
                     "tool use", "fine-tuning", "postgresql", "docker", "openai api", "semantic search"],
        "resp": [
            "Build LLM-powered agents that call tools, keep memory and return structured outputs",
            "Design retrieval pipelines (embeddings, hybrid search, re-ranking) over large document sets",
            "Create evaluation harnesses to measure and improve LLM output quality",
            "Iterate on prompts and guardrails to reduce hallucinations",
            "Ship AI features to production and keep latency and cost under control",
        ],
    },
    "data_eng": {
        "titles": {"intern": ["Data Engineering Intern"],
                   "junior": ["Data Engineer", "Analytics Engineer"],
                   "mid": ["Data Engineer II", "Senior Analytics Engineer"],
                   "senior": ["Senior Data Engineer", "Staff Data Engineer"],
                   "lead": ["Data Platform Lead"]},
        "core": ["sql", "python", "etl"],
        "optional": ["spark", "airflow", "dbt", "snowflake", "bigquery", "kafka", "aws", "postgresql",
                     "data modeling", "redshift", "docker"],
        "resp": [
            "Build reliable batch and streaming pipelines that feed analytics and ML",
            "Design warehouse schemas and dimensional models the whole company trusts",
            "Add data quality checks, lineage and alerting to every pipeline",
            "Tune slow SQL and reduce warehouse costs",
        ],
    },
    "devops": {
        "titles": {"intern": ["DevOps Intern", "Cloud Engineering Intern"],
                   "junior": ["DevOps Engineer", "Site Reliability Engineer I"],
                   "mid": ["Site Reliability Engineer", "Platform Engineer"],
                   "senior": ["Senior SRE", "Senior Platform Engineer"],
                   "lead": ["SRE Lead", "Head of Infrastructure"]},
        "core": ["linux", "docker", "ci/cd"],
        "optional": ["kubernetes", "terraform", "aws", "gcp", "prometheus", "grafana", "python", "ansible",
                     "bash", "azure", "helm"],
        "resp": [
            "Automate infrastructure with Terraform and keep environments reproducible",
            "Run Kubernetes clusters and improve deploy speed and safety",
            "Build observability: metrics, logs, traces and actionable alerts",
            "Lead incident response and write blameless post-mortems",
        ],
    },
    "mobile": {
        "titles": {"intern": ["Mobile Engineering Intern", "Android Intern"],
                   "junior": ["Android Engineer", "iOS Engineer", "Flutter Developer"],
                   "mid": ["Android Engineer II", "iOS Engineer II", "React Native Engineer"],
                   "senior": ["Senior Android Engineer", "Senior iOS Engineer"],
                   "lead": ["Mobile Lead"]},
        "core": ["mobile development", "git"],
        "optional": ["kotlin", "swift", "flutter", "react native", "android sdk", "ios", "jetpack compose",
                     "rest apis", "firebase", "dart", "swiftui"],
        "resp": [
            "Build smooth, offline-friendly mobile apps used by millions",
            "Improve app startup time, crash rate and battery usage",
            "Integrate with REST/GraphQL APIs and handle flaky networks gracefully",
            "Ship through staged rollouts and monitor crash-free sessions",
        ],
    },
    "qa": {
        "titles": {"intern": ["QA Intern", "SDET Intern"],
                   "junior": ["QA Engineer", "SDET I"],
                   "mid": ["SDET II", "Senior QA Engineer"],
                   "senior": ["Senior SDET", "QA Architect"],
                   "lead": ["QA Lead"]},
        "core": ["test automation", "sql"],
        "optional": ["selenium", "pytest", "cypress", "playwright", "java", "python", "postman", "ci/cd",
                     "api testing", "jenkins"],
        "resp": [
            "Build and maintain automated UI and API test suites",
            "Define test strategy and quality gates in CI",
            "Find, reproduce and triage bugs with engineers",
        ],
    },
    "security": {
        "titles": {"junior": ["Security Engineer", "Application Security Analyst"],
                   "mid": ["Security Engineer II", "Cloud Security Engineer"],
                   "senior": ["Senior Security Engineer"],
                   "lead": ["Security Lead"]},
        "core": ["application security", "linux"],
        "optional": ["penetration testing", "owasp", "aws", "python", "siem", "threat modeling", "iam",
                     "vulnerability management", "docker"],
        "resp": [
            "Review designs and code for security flaws and fix them with the owning teams",
            "Run vulnerability scans and drive remediation",
            "Build detections and respond to security incidents",
        ],
    },
    "pm": {
        "titles": {"intern": ["Product Management Intern"],
                   "junior": ["Associate Product Manager"],
                   "mid": ["Product Manager"],
                   "senior": ["Senior Product Manager"],
                   "lead": ["Group Product Manager", "Director of Product"]},
        "core": ["product management", "stakeholder management"],
        "optional": ["sql", "a/b testing", "roadmapping", "user research", "analytics", "agile", "jira", "figma"],
        "resp": [
            "Own the roadmap for a product area and write crisp PRDs",
            "Use data and user interviews to decide what to build next",
            "Partner with engineering and design to ship on time and measure outcomes",
        ],
    },
    "analyst": {
        "titles": {"intern": ["Data Analyst Intern", "Business Analyst Intern"],
                   "junior": ["Data Analyst", "Business Analyst"],
                   "mid": ["Senior Data Analyst", "Product Analyst"],
                   "senior": ["Lead Data Analyst", "Analytics Manager"],
                   "lead": ["Head of Analytics"]},
        "core": ["sql", "data analysis"],
        "optional": ["excel", "tableau", "power bi", "python", "looker", "statistics", "a/b testing", "pandas"],
        "resp": [
            "Answer business questions with SQL and clear, honest storytelling",
            "Build dashboards that teams actually open every morning",
            "Design and analyse experiments with product teams",
        ],
    },
    "design": {
        "titles": {"intern": ["Product Design Intern"],
                   "junior": ["UI/UX Designer", "Product Designer"],
                   "mid": ["Product Designer II", "Interaction Designer"],
                   "senior": ["Senior Product Designer"],
                   "lead": ["Design Lead"]},
        "core": ["ui design", "ux design", "figma"],
        "optional": ["user research", "prototyping", "design systems", "accessibility", "motion design"],
        "resp": [
            "Design end-to-end flows from research and wireframes to polished UI",
            "Maintain and extend our design system",
            "Run usability tests and iterate with engineers",
        ],
    },
}

FAMILY_WEIGHTS = {"backend": 20, "frontend": 11, "fullstack": 13, "ml": 9, "llm": 8, "data_eng": 7, "devops": 7,
                  "mobile": 6, "qa": 4, "security": 3, "pm": 5, "analyst": 5, "design": 3}

DOMAINS = [
    "a fintech platform serving first-time investors", "an AI-first career platform for students",
    "a B2B SaaS product for logistics teams", "a consumer app for food delivery", "a healthtech startup",
    "an edtech platform with 5M learners", "a developer-tools company", "an e-commerce marketplace",
    "a travel booking platform", "a climate and energy analytics company", "a gaming studio",
    "a cybersecurity product", "a HR-tech platform", "a payments infrastructure company",
]

INTRO = [
    "We are {company}, {domain}. ",
    "{company} is building {domain}, and we are growing fast. ",
    "Join {company}, {domain}, at an exciting stage. ",
]
CULTURE = [
    "You will have real ownership from day one and ship to production in your first week.",
    "We value curiosity, small teams and fast feedback loops over process.",
    "Expect mentorship from senior engineers and a strong written-docs culture.",
    "We are a small team where your work directly moves company metrics.",
]
LEVEL_REQ = {
    "intern": "Currently pursuing a degree in Computer Science or a related field. Strong fundamentals and a project you are proud of.",
    "junior": "0-2 years of experience. Strong fundamentals and the ability to learn new tools quickly.",
    "mid": "2-4 years of relevant experience shipping production systems.",
    "senior": "5+ years of experience with a track record of technical leadership on complex projects.",
    "lead": "8+ years of experience, including leading teams and driving multi-quarter technical or product strategy.",
}
SENIORITY_WEIGHTS = {"intern": 22, "junior": 28, "mid": 26, "senior": 18, "lead": 6}

# Salary bands by seniority, INR per year for India; (lo, hi) in lakhs. Interns: INR per month.
INR_LPA = {"junior": (6, 16), "mid": (14, 32), "senior": (30, 65), "lead": (55, 110)}
INTERN_MONTHLY_INR = (15000, 90000)
USD_K = {"junior": (70, 110), "mid": (100, 150), "senior": (140, 210), "lead": (190, 280)}
CURRENCY_FOR = {"United Kingdom": ("GBP", 0.8), "Germany": ("EUR", 0.9), "Singapore": ("SGD", 1.2)}


def wchoice(rng: random.Random, pairs):
    items, weights = zip(*pairs, strict=True)
    return rng.choices(items, weights=weights, k=1)[0]


def build_job(i: int, rng: random.Random) -> dict:
    fam_name = wchoice(rng, FAMILY_WEIGHTS.items())
    fam = FAMILIES[fam_name]
    seniority = wchoice(rng, [(s, w) for s, w in SENIORITY_WEIGHTS.items() if s in fam["titles"]])
    title = rng.choice(fam["titles"][seniority])
    company = rng.choice(COMPANIES)
    city, country, _ = wchoice(rng, [((c, k, w), w) for c, k, w in LOCATIONS])

    optional_n = rng.randint(2, min(6, len(fam["optional"])))
    skills = list(dict.fromkeys(fam["core"] + rng.sample(fam["optional"], optional_n)))

    if city.startswith("Remote"):
        work_mode = "remote"
    else:
        work_mode = rng.choices(["onsite", "hybrid", "remote"], weights=[40, 45, 15])[0]
    job_type = "internship" if seniority == "intern" else rng.choices(
        ["full_time", "contract", "part_time"], weights=[90, 8, 2])[0]

    # Salary (~75% disclosed)
    sal = {"salary_min": None, "salary_max": None, "salary_currency": None, "salary_period": None}
    if rng.random() < 0.75:
        if country == "India":
            if seniority == "intern":
                lo = rng.randrange(INTERN_MONTHLY_INR[0], INTERN_MONTHLY_INR[1] - 10000, 5000)
                sal = dict(salary_min=lo, salary_max=lo + rng.choice([5000, 10000, 15000]),
                           salary_currency="INR", salary_period="month")
            else:
                lo_l, hi_l = INR_LPA[seniority]
                lo = rng.randint(lo_l, hi_l - 3)
                sal = dict(salary_min=lo * 100000, salary_max=(lo + rng.randint(3, 10)) * 100000,
                           salary_currency="INR", salary_period="year")
        elif seniority != "intern":
            lo_k, hi_k = USD_K[seniority]
            cur, fx = CURRENCY_FOR.get(country, ("USD", 1.0))
            lo = int(rng.randint(lo_k, hi_k - 15) * fx) * 1000
            sal = dict(salary_min=lo, salary_max=lo + int(rng.randint(10, 35) * fx) * 1000,
                       salary_currency=cur, salary_period="year")

    resp = rng.sample(fam["resp"], k=min(len(fam["resp"]), rng.randint(3, 5)))
    must, nice = skills[: max(3, len(skills) // 2 + 1)], skills[max(3, len(skills) // 2 + 1):]
    domain = rng.choice(DOMAINS)
    desc = (
        rng.choice(INTRO).format(company=company, domain=domain) + rng.choice(CULTURE) + "\n\n"
        "What you'll do:\n" + "\n".join(f"- {r}." for r in resp) + "\n\n"
        f"What we're looking for:\n- {LEVEL_REQ[seniority]}\n"
        f"- Hands-on experience with {', '.join(must)}.\n"
        + (f"- Nice to have: {', '.join(nice)}.\n" if nice else "")
        + f"\nLocation: {city} ({work_mode}). "
        + ("This is a time-bound internship with a pre-placement offer (PPO) for strong performers." if seniority == "intern" and rng.random() < 0.5 else "")
    ).strip()

    slug = company.lower().replace(" ", "-")
    return {
        "external_id": f"seed-{i:04d}",
        "company": company,
        "title": title,
        "description": desc,
        "location": city,
        "country": country,
        "work_mode": work_mode,
        "job_type": job_type,
        "seniority": seniority,
        **sal,
        "url": f"https://careers.example.com/{slug}/jobs/seed-{i:04d}",
        "posted_days_ago": rng.randint(0, 45),
        "skills": skills,
        "role_family": fam_name,  # ground truth, used by the eval harness
    }


def main() -> None:
    rng = random.Random(42)
    jobs = [build_job(i, rng) for i in range(1, N_JOBS + 1)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    fams = {}
    for j in jobs:
        fams[j["role_family"]] = fams.get(j["role_family"], 0) + 1
    print(f"wrote {len(jobs)} jobs to {OUT}\nfamilies: {fams}")


if __name__ == "__main__":
    main()
