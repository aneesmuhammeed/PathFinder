import pytest

from app.ingest.skills import canonicalize, extract_skills, is_known
from app.ingest.text import (
    html_to_text,
    infer_country,
    infer_job_type,
    infer_seniority,
    is_tech_role,
    looks_english,
    parse_salary,
)


class TestSkills:
    def test_aliases_canonicalize(self):
        assert canonicalize("Postgres") == "postgresql"
        assert canonicalize("  K8s ") == "kubernetes"
        assert canonicalize("GOLANG") == "go"
        assert canonicalize("totally-unknown-skill") == "totally-unknown-skill"

    def test_extracts_canonical_in_order(self):
        text = "We use Python, FastAPI and Postgres on AWS with k8s."
        assert extract_skills(text) == ["python", "fastapi", "postgresql", "aws", "kubernetes"]

    def test_java_does_not_match_javascript_and_vice_versa(self):
        assert extract_skills("JavaScript developer") == ["javascript"]
        assert extract_skills("Java developer") == ["java"]

    def test_longest_alias_wins(self):
        assert extract_skills("React Native apps") == ["react native"]
        assert extract_skills("Node.js and React") == ["node.js", "react"]

    def test_special_character_skills(self):
        assert set(extract_skills("C++, C# and .NET experience")) == {"c++", "c#", ".net"}

    def test_ambiguous_words_do_not_match_bare(self):
        assert extract_skills("Ready to go the extra mile, express interest today") == []
        assert extract_skills("Services written in Golang") == ["go"]

    def test_sentence_end_punctuation(self):
        assert extract_skills("Strong in SQL. Also Python.") == ["sql", "python"]

    def test_limit_prefers_title_then_frequency(self):
        stuffed = "java java java python python go golang rust ruby php scala kotlin swift dart sql"
        # Title skill wins even though it appears once; then most frequent; stable otherwise.
        top = extract_skills(stuffed + " react", title="Senior React Engineer", limit=3)
        assert top == ["react", "java", "python"]
        assert extract_skills("python sql", limit=5) == ["python", "sql"]  # under the cap: untouched

    def test_is_known(self):
        assert is_known("pgvector") is True  # alias of 'vector databases'
        assert is_known("postgres") is True
        assert is_known("fullstack") is False


class TestHtml:
    def test_strips_tags_keeps_structure(self):
        html = "<h3>Role</h3><p>Build &amp; ship</p><ul><li>One</li><li>Two</li></ul><script>x()</script>"
        out = html_to_text(html)
        assert "<" not in out and "x()" not in out
        assert "Build & ship" in out
        assert "- One" in out and "- Two" in out

    def test_plain_text_passthrough(self):
        assert html_to_text("just text &amp; more") == "just text & more"


class TestSalary:
    @pytest.mark.parametrize("raw,expected", [
        ("$170k - $200k", (170000, 200000, "USD", "year")),
        ("$80,000 - $120,000", (80000, 120000, "USD", "year")),
        ("€50-60k", (50000, 60000, "EUR", "year")),
        ("£3,000 - £4,000 per month", (3000, 4000, "GBP", "month")),
    ])
    def test_parses(self, raw, expected):
        assert parse_salary(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "$14 - $20/hour", "competitive", "5-10 years", "$35,3k- $52k"])
    def test_refuses_when_unsure(self, raw):
        assert parse_salary(raw) is None


class TestClassifiers:
    @pytest.mark.parametrize("title,expected", [
        ("Software Engineering Intern", "intern"),
        ("Werkstudent IT", "intern"),
        ("Senior Backend Engineer", "senior"),
        ("Staff Software Engineer", "senior"),
        ("Engineering Manager, Platform", "lead"),
        ("Principal Engineer", "lead"),
        ("Junior Data Analyst", "junior"),
        ("Product Manager", "mid"),
        ("Senior Product Manager", "senior"),
        ("Backend Engineer", "mid"),
    ])
    def test_seniority(self, title, expected):
        assert infer_seniority(title) == expected

    def test_seniority_uses_hints(self):
        assert infer_seniority("Developer", "entry") == "junior"

    def test_job_type(self):
        assert infer_job_type("SDE Intern") == "internship"
        assert infer_job_type("Dev", "Contract") == "contract"
        assert infer_job_type("Dev", "Teilzeit") == "part_time"
        assert infer_job_type("Dev") == "full_time"

    def test_is_tech_role(self):
        assert is_tech_role("Backend Engineer", [])
        assert not is_tech_role("Pflegefachkraft", [])
        for t in ("Go-To-Market Engineer", "Senior Sales Engineer", "Marketing Automation Engineer"):
            assert not is_tech_role(t, ["python", "sql", "aws"])  # engineer-in-title but not an engineering job
        assert is_tech_role("Operations Associate", ["python", "sql", "excel"])

    def test_looks_english(self):
        en = "We are looking for an engineer and you will work with our team to build the platform. " * 3
        de = "Wir suchen einen Entwickler und Sie arbeiten mit unserem Team für die Plattform und der Kunden. " * 3
        assert looks_english(en) and not looks_english(de)

    def test_country(self):
        assert infer_country("Bengaluru, Karnataka") == "India"
        assert infer_country("Berlin") == "Germany"
        assert infer_country("Worldwide") is None
