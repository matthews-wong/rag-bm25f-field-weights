import unittest

from bm25f.corpus import load_docs
from bm25f.evaluate import FIXTURES, evaluate, load_queries
from bm25f.index import BM25FIndex


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.docs = load_docs(FIXTURES / "docs.json")
        self.queries = load_queries(FIXTURES / "queries.json")

    def test_every_label_points_at_a_real_document(self):
        for case in self.queries:
            self.assertIn(case["relevant"], self.docs, case["query"])

    def test_metrics_are_bounded(self):
        recall, mrr = evaluate(BM25FIndex(self.docs), self.queries)
        self.assertTrue(0 < mrr <= recall <= 1)

    def test_title_weighting_does_not_lose_to_flat_baseline(self):
        flat = {i: {"text": f"{d['title']} {d['body']}"} for i, d in self.docs.items()}
        _, baseline = evaluate(BM25FIndex(flat, weights={"text": 1.0}), self.queries)
        _, weighted = evaluate(BM25FIndex(self.docs), self.queries)
        self.assertGreaterEqual(weighted, baseline)

    def test_missing_relevant_document_scores_zero(self):
        recall, mrr = evaluate(BM25FIndex(self.docs), [{"query": "zzz", "relevant": "x"}])
        self.assertEqual((recall, mrr), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
