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
python3 -m bm25f "tls certificate renewal" --corpus fixtures/docs.json
python3 -m bm25f "tls certificate renewal" --title-weight 1 --top 3
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

## Fixture results

`python3 -m bm25f.evaluate` runs every variant over the checked-in
fixtures (`fixtures/docs.json`, `fixtures/queries.json`). These are
fixture numbers on a tiny corpus, not a benchmark.

```
fixtures: 12 docs, 16 queries
run                           recall@3     MRR
plain BM25 (concatenated)        1.000   0.906
BM25F title x1                   1.000   0.938
BM25F title x2                   1.000   0.938
BM25F title x3                   1.000   0.938
BM25F title x5                   1.000   0.938
```

Title weighting lifts MRR a little (a title hit moves the intended
document up) and costs nothing on recall@3 here. With only 16 queries one
reordered result changes MRR by about 0.03, so treat the gap as a sanity
check, not evidence. Queries whose answer lives only in a body are in the
set on purpose; weights above 1 must not bury them.

There is no stemming, so `rotate` does not match `rotating`.

## Validate

```
make validate
```

Runs the unit tests and the fixture eval.
