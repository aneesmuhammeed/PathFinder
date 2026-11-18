"""API tests. The app's DB dependency is bound to the per-test rolled-back session."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder_dep
from app.db.session import get_db
from app.main import create_app
from tests.helpers import EMB, load, nj


@pytest.fixture()
def api(db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: EMB
    return TestClient(app)


@pytest.fixture()
def ids(db):
    return load(db, [
        nj("py", "Python Backend Engineer", ["python", "sql"], seniority="junior"),
        nj("jv", "Java Backend Engineer", ["java"], seniority="senior"),
    ])


def test_search_endpoint_returns_explainable_results(api, ids):
    r = api.post("/search", json={"query": "backend engineer", "profile": {"skills": ["python"]}, "limit": 5})
    assert r.status_code == 200
    body = r.json()
    top = body["results"][0]
    assert top["job"]["id"] == ids["py"] and body["personalised"]
    assert {"feature", "value", "weight", "contribution", "detail"} <= set(top["components"][0])
    assert top["job"]["company"] and top["job"]["skills"] == ["python", "sql"]


def test_search_validation_errors(api):
    assert api.post("/search", json={"query": ""}).status_code == 422  # nothing to search on
    assert api.post("/search", json={"query": "x", "limit": 500}).status_code == 422
    assert api.post("/search", json={"query": "x", "filters": {"seniority": ["wizard"]}}).status_code == 422
    assert api.post("/search", json={"query": "x", "user_id": str(uuid.uuid4())}).status_code == 404


def test_get_job(api, ids):
    r = api.get(f"/jobs/{ids['py']}")
    assert r.status_code == 200 and r.json()["title"] == "Python Backend Engineer"
    assert api.get("/jobs/99999999").status_code == 404


def test_user_profile_feedback_flow(api, ids):
    uid = api.post("/users", json={}).json()["id"]

    # Profile memory: lists merge across updates, scalars replace.
    api.patch(f"/users/{uid}/profile", json={"skills": ["Python"], "seniority": "junior"})
    p = api.patch(f"/users/{uid}/profile", json={"skills": ["Postgres"], "seniority": "mid"}).json()
    assert p["skills"] == ["python", "postgresql"] and p["seniority"] == "mid"
    p = api.patch(f"/users/{uid}/profile", json={"remove_skills": ["python"]}).json()
    assert p["skills"] == ["postgresql"]
    assert api.get(f"/users/{uid}/profile").json()["seniority"] == "mid"

    # Feedback: dismissed jobs vanish from search; saved ones are listed.
    assert api.post(f"/users/{uid}/feedback", json={"job_id": ids["jv"], "action": "dismissed"}).status_code == 204
    assert api.post(f"/users/{uid}/feedback", json={"job_id": ids["py"], "action": "saved"}).status_code == 204
    res = api.post("/search", json={"query": "backend engineer", "user_id": uid}).json()
    assert [r["job"]["id"] for r in res["results"]] == [ids["py"]] and res["results"][0]["saved"]
    assert [j["id"] for j in api.get(f"/users/{uid}/saved").json()] == [ids["py"]]

    api.delete(f"/users/{uid}/feedback/saved/{ids['py']}")
    assert api.get(f"/users/{uid}/saved").json() == []


def test_profile_can_be_wiped_and_the_agent_forgets(api, ids):
    uid = api.post("/users", json={}).json()["id"]
    api.patch(f"/users/{uid}/profile", json={"skills": ["python"], "seniority": "mid", "min_salary": 900000, "summary": "note"})
    wiped = api.delete(f"/users/{uid}/profile")
    assert wiped.status_code == 200
    assert wiped.json() == {"skills": [], "target_roles": [], "preferred_locations": [], "seniority": None,
                            "work_mode": None, "job_type": None, "min_salary": None}
    assert api.get(f"/users/{uid}/profile").json()["skills"] == []
    assert api.delete(f"/users/{uuid.uuid4()}/profile").status_code == 404


def test_scalar_profile_fields_can_be_cleared_with_null(api):
    uid = api.post("/users", json={}).json()["id"]
    api.patch(f"/users/{uid}/profile", json={"work_mode": "remote", "min_salary": 500000})
    p = api.patch(f"/users/{uid}/profile", json={"work_mode": None}).json()
    assert p["work_mode"] is None and p["min_salary"] == 500000  # only the cleared field changed


def test_user_endpoint_errors(api, ids):
    ghost = uuid.uuid4()
    assert api.get(f"/users/{ghost}/profile").status_code == 404
    assert api.patch(f"/users/{ghost}/profile", json={"skills": ["x"]}).status_code == 404
    uid = api.post("/users", json={}).json()["id"]
    assert api.patch(f"/users/{uid}/profile", json={"seniority": "wizard"}).status_code == 422
    assert api.post(f"/users/{uid}/feedback", json={"job_id": 987654321, "action": "saved"}).status_code == 404
    assert api.post(f"/users/{uid}/feedback", json={"job_id": ids["py"], "action": "love"}).status_code == 422
