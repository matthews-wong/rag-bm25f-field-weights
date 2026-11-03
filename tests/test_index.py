import unittest

from bm25f.index import BM25FIndex

DOCS = {
    "a": {"title": "Rollback guide", "body": "steps to undo a release"},
    "b": {"title": "Release notes", "body": "rollback is mentioned once in this long body of text"},
    "c": {"title": "Unrelated", "body": "nothing relevant here"},
}


class IndexTest(unittest.TestCase):
    def test_title_match_outranks_body_match(self):
        ranked = [d for d, _ in BM25FIndex(DOCS).search("rollback")]
        self.assertEqual(ranked[:2], ["a", "b"])

    def test_equal_weights_remove_title_preference(self):
        docs = {
            "t": {"title": "alpha", "body": "x y z"},
            "u": {"title": "x", "body": "alpha y z"},
        }
        index = BM25FIndex(docs, weights={"title": 1.0, "body": 1.0})
        scores = dict(index.search("alpha"))
        self.assertAlmostEqual(scores["t"], scores["u"])

    def test_zero_score_documents_are_dropped(self):
        self.assertNotIn("c", dict(BM25FIndex(DOCS).search("rollback")))

    def test_ties_break_on_id(self):
        docs = {"z": {"title": "same", "body": ""}, "y": {"title": "same", "body": ""}}
        self.assertEqual([d for d, _ in BM25FIndex(docs).search("same")], ["y", "z"])

    def test_missing_field_counts_as_empty(self):
        index = BM25FIndex({"a": {"title": "only title"}, "b": {"title": "x", "body": "y"}})
        self.assertEqual(index.search("title")[0][0], "a")

    def test_repeated_query_terms_count_once(self):
        index = BM25FIndex(DOCS)
        self.assertEqual(index.search("rollback rollback"), index.search("rollback"))

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            BM25FIndex({})

    def test_negative_weight_names_the_field(self):
        with self.assertRaisesRegex(ValueError, "'title'.*-1"):
            BM25FIndex(DOCS, weights={"title": -1})

    def test_b_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "1.5"):
            BM25FIndex(DOCS, b=1.5)


if __name__ == "__main__":
    unittest.main()
