"""BM25F over documents made of named text fields."""

import math
from collections import Counter

from .tokenize import tokenize

K1 = 1.2
DEFAULT_B = 0.75
DEFAULT_WEIGHTS = {"title": 3.0, "body": 1.0}


class BM25FIndex:
    def __init__(
        self,
        docs: dict[str, dict[str, str]],
        weights: dict[str, float] | None = None,
        b: float = DEFAULT_B,
    ) -> None:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"field weight for {name!r} must be >= 0, got {weight}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")
        self.b = b
        self.tf: dict[str, dict[str, Counter[str]]] = {}
        for doc_id, fields in docs.items():
            self.tf[doc_id] = {f: Counter(tokenize(fields.get(f, ""))) for f in self.weights}
        self.avg_length = {
            f: (sum(sum(d[f].values()) for d in self.tf.values()) / len(docs)) or 1.0
            for f in self.weights
        }
        df: Counter[str] = Counter()
        for fields in self.tf.values():
            seen: set[str] = set()
            for counter in fields.values():
                seen.update(counter)
            df.update(seen)
        total = len(docs)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _weighted_tf(self, doc_id: str, term: str) -> float:
        total = 0.0
        for name, weight in self.weights.items():
            counter = self.tf[doc_id][name]
            freq = counter.get(term, 0)
            if not freq:
                continue
            length = sum(counter.values())
            norm = 1 - self.b + self.b * length / self.avg_length[name]
            total += weight * freq / norm
        return total

    def score(self, doc_id: str, terms: list[str]) -> float:
        total = 0.0
        for term in terms:
            tf = self._weighted_tf(doc_id, term)
            if tf:
                total += self.idf[term] * tf / (K1 + tf)
        return total

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        scored = [(doc_id, self.score(doc_id, terms)) for doc_id in self.tf]
        # Ties break on id so rankings are reproducible.
        ranked = sorted((s for s in scored if s[1] > 0), key=lambda s: (-s[1], s[0]))
        return ranked[:limit]
