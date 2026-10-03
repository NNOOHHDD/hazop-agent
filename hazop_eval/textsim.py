"""Text similarity for the embedding-based matcher.

tfidf  - local, offline, lexical (word overlap weighted by rarity); no API key.
openai - OpenAI text-embedding-3-small (OPENAI_API_KEY); semantic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""a an and are as at be by for from in into is it its of on or that the to with without due
because caused cause leads lead leading resulting result results possible potential""".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOP and len(w) > 1]


def _norm(v: dict) -> dict:
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / n for k, x in v.items()}


def tfidf_vectors(texts: list[str]) -> list[dict]:
    docs = [Counter(tokens(t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    return [_norm({w: c * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in d.items()}) for d in docs]


def cosine_sparse(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


def openai_vectors(texts: list[str], model: str = "text-embedding-3-small") -> list[dict]:
    import openai
    client = openai.OpenAI(max_retries=4)
    out = []
    for i in range(0, len(texts), 256):
        resp = client.embeddings.create(model=model, input=texts[i:i + 256])
        out += [_norm(dict(enumerate(d.embedding))) for d in resp.data]
    return out


def similarity_matrix(left: list[str], right: list[str], backend: str = "tfidf") -> list[list[float]]:
    vecs = (openai_vectors if backend == "openai" else tfidf_vectors)(left + right)
    lv, rv = vecs[:len(left)], vecs[len(left):]
    return [[cosine_sparse(a, b) for b in rv] for a in lv]


DEFAULT_THRESHOLDS = {"tfidf": 0.30, "openai": 0.55}
