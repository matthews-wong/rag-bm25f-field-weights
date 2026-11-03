"""Command line search: python3 -m bm25f "query" --corpus fixtures/docs.json."""

import argparse

from .corpus import load_docs
from .index import DEFAULT_WEIGHTS, BM25FIndex


def main() -> None:
    parser = argparse.ArgumentParser(prog="bm25f", description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--corpus", default="fixtures/docs.json")
    parser.add_argument("--title-weight", type=float, default=DEFAULT_WEIGHTS["title"])
    parser.add_argument("--body-weight", type=float, default=DEFAULT_WEIGHTS["body"])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    docs = load_docs(args.corpus)
    index = BM25FIndex(docs, weights={"title": args.title_weight, "body": args.body_weight})
    results = index.search(args.query, limit=args.top)
    if not results:
        print(f"no documents match {args.query!r}")
        return
    for doc_id, score in results:
        print(f"{score:6.3f}  {doc_id}  {docs[doc_id]['title']}")


if __name__ == "__main__":
    main()
