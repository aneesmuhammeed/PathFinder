"""Embedding backends behind one interface.

* ``HashEmbedder``: feature hashing of unigrams + bigrams. Deterministic, offline, keyless, no
  downloads. It is *lexical* similarity in a vector space, not true semantics, and exists so tests,
  CI and the offline demo are reproducible. Never present it as semantic search.
* ``FastEmbedEmbedder``: real local semantic model (BAAI/bge-small-en-v1.5 via ONNX).
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from app.core.config import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./-]*")
_STOP = frozenset(
    "a an the and or of to in for on with at by from as is are be we you our your will this that it its".split()
)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models use a different prefix for queries)."""


class HashEmbedder:
    name = "hash"
    # Document recipe chosen by a grid over (description chars x title repeats), scored on the pipeline we
    # actually ship (hybrid RRF) with the dev split, then confirmed on the held-out test split; see EVALS.md.
    # Repeating the title gives the role words more weight in the hashed vector. History worth knowing: an earlier
    # version truncated the description to 300 chars because that fixed vector-ONLY retrieval on 13 short role
    # queries; on the broader eval set it was a net loss for hybrid search (test nDCG 0.871 vs 0.890), so it was reverted.
    doc_params = {"description_chars": 2000, "title_repeats": 2}
    # Vote of this embedder's ranking in hybrid fusion (keyword = 1.0). A lexical hash vector is much noisier than
    # keyword search on short queries, so it gets a small vote; a real semantic model should start at 1.0 and be
    # re-tuned. Chosen by a two-part rule in EVALS.md 2.2 (near-best on dev AND the known "frontend engineer, remote"
    # failure must reach nDCG >= 0.75): dev nDCG 0.922 -> 0.929, that failure 0.38 -> 0.76. Test is flat.
    rrf_weight = 0.35

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _slot(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def _one(self, text: str) -> list[float]:
        toks = [t.strip(".-/") for t in _TOKEN.findall(text.lower())]
        toks = [t for t in toks if t and t not in _STOP]
        vec = [0.0] * self.dim
        counts: dict[str, float] = {}
        for t in toks:
            counts[t] = counts.get(t, 0.0) + 1.0
        for a, b in zip(toks, toks[1:], strict=False):  # adjacent word pairs (bigrams)
            key = f"{a} {b}"
            counts[key] = counts.get(key, 0.0) + 0.5
        for tok, c in counts.items():
            idx, sign = self._slot(tok)
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("fastembed not installed: pip install -r requirements-ml.txt") from exc
        self.dim = dim
        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - needs model download
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed([BGE_QUERY_PREFIX + text])[0]


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedder == "fastembed":
        return FastEmbedEmbedder(s.fastembed_model, s.embedding_dim)
    if s.embedder == "hash":
        return HashEmbedder(s.embedding_dim)
    raise ValueError(f"unknown EMBEDDER={s.embedder!r} (expected 'hash' or 'fastembed')")


def job_embedding_text(
    *, title: str, company: str, location: str | None, seniority: str, job_type: str,
    work_mode: str, skills: Sequence[str], description: str,
    description_chars: int = 2000, title_repeats: int = 1,
) -> str:
    """The exact text a job is embedded from. Keep stable: changing it means re-embedding everything
    (`python -m app.cli reembed`). `description_chars` / `title_repeats` come from the embedder's `doc_params`."""
    head = f"{title} at {company}. {seniority} {job_type.replace('_', ' ')}, {work_mode}"
    if location:
        head += f", {location}"
    lead = f"{title}. " * (title_repeats - 1)
    return f"{lead}{head}. Skills: {', '.join(skills)}.\n{description[:description_chars]}"
