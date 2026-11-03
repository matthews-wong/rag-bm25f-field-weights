"""Lowercase word tokenizer shared by indexing and querying."""

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
