"""Compare plain BM25 (title and body concatenated) with BM25F on labelled fixtures."""

from pathlib import Path

from .corpus import load_docs, load_queries
from .index import BM25FIndex

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
K = 3


def evaluate(index: BM25FIndex, queries: list[dict[str, str]], k: int = K) -> tuple[float, float]:
    """Return (recall@k, MRR) for single-relevant-document queries."""
    hits = 0
    reciprocal = 0.0
    for case in queries:
        ranked = [doc_id for doc_id, _ in index.search(case["query"], limit=len(index.tf))]
        if case["relevant"] in ranked[:k]:
            hits += 1
        if case["relevant"] in ranked:
            reciprocal += 1 / (ranked.index(case["relevant"]) + 1)
    return hits / len(queries), reciprocal / len(queries)


def main() -> None:
    docs = load_docs(FIXTURES / "docs.json")
    queries = load_queries(FIXTURES / "queries.json")
    # Plain BM25 baseline: one concatenated field, so a title hit is worth no more than a body hit.
    flat = {i: {"text": f"{d['title']} {d['body']}"} for i, d in docs.items()}
    runs = [("plain BM25 (concatenated)", BM25FIndex(flat, weights={"text": 1.0}))]
    for title_weight in (1.0, 2.0, 3.0, 5.0):
        weights = {"title": title_weight, "body": 1.0}
        runs.append((f"BM25F title x{title_weight:g}", BM25FIndex(docs, weights=weights)))
    print(f"fixtures: {len(docs)} docs, {len(queries)} queries")
    print(f"{'run':<28}{f'recall@{K}':>10}{'MRR':>8}")
    for name, index in runs:
        recall, mrr = evaluate(index, queries)
        print(f"{name:<28}{recall:>10.3f}{mrr:>8.3f}")


if __name__ == "__main__":
    main()
