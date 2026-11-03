"""Load documents and labelled queries from JSON fixtures."""

import json
from pathlib import Path


def load_docs(path: str | Path) -> dict[str, dict[str, str]]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: dict[str, dict[str, str]] = {}
    for position, item in enumerate(items):
        if "id" not in item:
            raise ValueError(f"{path}: document at position {position} has no 'id'")
        if item["id"] in docs:
            raise ValueError(f"{path}: duplicate document id {item['id']!r}")
        docs[item["id"]] = {k: v for k, v in item.items() if k != "id"}
    return docs


def load_queries(path: str | Path) -> list[dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
