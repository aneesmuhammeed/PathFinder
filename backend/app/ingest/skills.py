"""Skill taxonomy: canonical names, categories, aliases, and text extraction.

Format per line: canonical | category | comma-separated aliases. A trailing "!" on the canonical name
means the bare name is too ambiguous to match in free text ("go", "express"); only its aliases match.
"""

import re

_TAXONOMY = """
python|language|
java|language|
javascript|language|js,ecmascript
typescript|language|ts
go!|language|golang
rust|language|
c++|language|cpp
c#|language|csharp
.net|backend|dotnet,asp.net
php|language|
ruby|language|
rails|backend|ruby on rails
kotlin|mobile|
swift|mobile|
dart|mobile|
scala|language|
bash|devops|shell scripting
sql|data|
postgresql|data|postgres
mysql|data|
sqlite|data|
mongodb|data|mongo
redis|data|
elasticsearch|data|elastic search,opensearch
kafka|data|apache kafka
rabbitmq|data|
fastapi|backend|
django|backend|
flask|backend|
node.js|backend|nodejs,node
express!|backend|express.js,expressjs
spring boot|backend|springboot
rest apis|backend|rest api,restful,restful apis
graphql|backend|
grpc|backend|
microservices|backend|microservice
celery|backend|
react|frontend|reactjs,react.js
next.js|frontend|nextjs
vue.js|frontend|vue,vuejs
angular|frontend|angularjs
redux|frontend|
html|frontend|html5
css|frontend|css3
tailwind css|frontend|tailwind,tailwindcss
webpack|frontend|
jest|testing|
accessibility|frontend|a11y,wcag
react native|mobile|
flutter|mobile|
android sdk|mobile|android
ios|mobile|
jetpack compose|mobile|
swiftui|mobile|
firebase|mobile|
mobile development|mobile|mobile app development
docker|devops|
kubernetes|devops|k8s
terraform|devops|
ansible|devops|
helm|devops|
jenkins|devops|
ci/cd|devops|cicd,continuous integration
linux|devops|
aws|cloud|amazon web services
gcp|cloud|google cloud,google cloud platform
azure|cloud|microsoft azure
prometheus|devops|
grafana|devops|
machine learning|ml|ml
deep learning|ml|
nlp|ml|natural language processing
computer vision|ml|
pytorch|ml|
tensorflow|ml|
scikit-learn|ml|sklearn,scikit learn
pandas|ml|
numpy|ml|
xgboost|ml|
mlops|ml|
feature engineering|ml|
statistics|ml|statistical analysis
a/b testing|ml|ab testing,a/b tests,experimentation
llm|llm|llms,large language models,large language model
generative ai|llm|genai,gen ai
prompt engineering|llm|prompting
rag|llm|retrieval augmented generation,retrieval-augmented generation
embeddings|llm|embedding,text embeddings
vector databases|llm|vector database,vector db,pgvector,pinecone,weaviate,faiss
langchain|llm|llamaindex
evals|llm|llm evaluation,llm evals
tool use|llm|function calling,tool calling
fine-tuning|llm|finetuning,fine tuning
openai api|llm|openai
hugging face|llm|huggingface,transformers
semantic search|llm|vector search
spark|data|apache spark,pyspark
airflow|data|apache airflow
dbt|data|
snowflake|data|
bigquery|data|
redshift|data|
etl|data|elt,data pipelines,data pipeline
data modeling|data|data modelling
data analysis|data|data analytics
excel|data|ms excel,microsoft excel
tableau|data|
power bi|data|powerbi
looker|data|
analytics|data|
selenium|testing|
pytest|testing|
cypress|testing|
playwright|testing|
postman|testing|
test automation|testing|automation testing,automated testing
api testing|testing|
application security|security|appsec
penetration testing|security|pentesting,pen testing
owasp|security|
siem|security|
threat modeling|security|
iam|security|
vulnerability management|security|
git|general|github,gitlab
agile|product|scrum
jira|product|
product management|product|product manager
stakeholder management|product|
roadmapping|product|product roadmap
user research|design|
ui design|design|ui/ux
ux design|design|
figma|design|
prototyping|design|
design systems|design|design system
motion design|design|
"""

CATEGORY: dict[str, str] = {}
_ALIAS_TO_CANONICAL: dict[str, str] = {}
_MATCHABLE: list[str] = []


def _load() -> None:
    for line in _TAXONOMY.strip().splitlines():
        name, category, aliases = (p.strip() for p in line.split("|"))
        ambiguous = name.endswith("!")
        canonical = name.rstrip("!")
        CATEGORY[canonical] = category
        _ALIAS_TO_CANONICAL[canonical] = canonical
        if not ambiguous:
            _MATCHABLE.append(canonical)
        for alias in filter(None, (a.strip() for a in aliases.split(","))):
            _ALIAS_TO_CANONICAL[alias] = canonical
            _MATCHABLE.append(alias)


_load()

# Longest first so "react native" wins over "react", "node.js" over "node".
_PATTERN = re.compile(
    r"(?<![a-z0-9+#])(" + "|".join(re.escape(a) for a in sorted(set(_MATCHABLE), key=len, reverse=True)) + r")(?![a-z0-9+#]|\.[a-z0-9])"
)


def canonicalize(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    return _ALIAS_TO_CANONICAL.get(n, n)


def is_known(name: str) -> bool:
    return canonicalize(name) in CATEGORY


def extract_skills(text: str, *, title: str = "", limit: int | None = None) -> list[str]:
    """Canonical skills mentioned in free text.

    Default: every skill, in order of first appearance. With `limit`, keep only the most relevant:
    skills named in the title first, then by how often they're mentioned. This stops keyword-stuffed
    postings (an ad that lists every technology) from matching everything.
    """
    counts: dict[str, int] = {}
    for m in _PATTERN.finditer(text.lower()):
        c = _ALIAS_TO_CANONICAL[m.group(1)]
        counts[c] = counts.get(c, 0) + 1
    if limit is None or len(counts) <= limit:
        return list(counts)
    in_title = set(extract_skills(title)) if title else set()
    order = {c: i for i, c in enumerate(counts)}
    ranked = sorted(counts, key=lambda c: (c not in in_title, -counts[c], order[c]))
    return ranked[:limit]
