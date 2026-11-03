# rag-bm25f-field-weights

Field-weighted lexical retrieval. Documents have a `title` and a `body`;
BM25F scores them as separate fields with their own weight and length
normalisation, instead of gluing the text together and running plain BM25.

The point: a query that matches a title is usually a stronger signal than
the same words buried in a body. Plain BM25 over concatenated text cannot
express that. BM25F can.

Pure Python 3.12 standard library. No model downloads, nothing to install.

## Usage

```
python3 -m bm25f "rollback a failed deploy" --corpus fixtures/docs.json
python3 -m bm25f "rollback a failed deploy" --title-weight 1 --top 3
```

## Scoring

For each query term, per-field term frequencies are normalised by field
length, weighted, and summed before saturation:

```
tf'(t, d) = sum over fields f of  w_f * tf(t, d, f) / (1 - b_f + b_f * len_f / avglen_f)
score(d)  = sum over terms t of   idf(t) * tf'(t, d) / (k1 + tf'(t, d))
```

`idf` uses document frequency over the whole document (any field), so a
term is not counted twice just because it appears in title and body.

## Validate

```
make validate
```

Runs the unit tests and the fixture eval.
