import math

import pytest

from app.search.embedder import HashEmbedder, get_embedder, job_embedding_text


def cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_deterministic_unit_norm_and_dim():
    e = HashEmbedder(384)
    a, b = e.embed(["python backend engineer"])[0], e.embed(["python backend engineer"])[0]
    assert a == b and len(a) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_related_text_scores_higher_than_unrelated():
    e = HashEmbedder(384)
    q = e.embed_query("python backend engineer postgres")
    close, far = e.embed(["Backend Engineer. Skills: python, postgresql, fastapi", "Product Designer. Skills: figma, ux research"])
    assert cos(q, close) > cos(q, far)


def test_empty_text_does_not_crash():
    assert len(HashEmbedder(384).embed([""])[0]) == 384


def test_factory_rejects_unknown_backend():
    from app.core.config import Settings

    with pytest.raises(ValueError):
        get_embedder(Settings(embedder="nope"))
    assert get_embedder(Settings(embedder="hash")).name == "hash"


def test_job_text_is_stable_and_truncates_description():
    kw = dict(title="T", company="C", location="Pune", seniority="mid", job_type="full_time", work_mode="hybrid",
              skills=["python"], description="x" * 5000)
    t = job_embedding_text(**kw)
    assert t == job_embedding_text(**kw)
    assert t.startswith("T at C. mid full time, hybrid, Pune. Skills: python.")
    assert len(t) < 2200
