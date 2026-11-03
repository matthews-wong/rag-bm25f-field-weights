import json
import tempfile
import unittest
from pathlib import Path

from bm25f.corpus import load_docs


class LoadDocsTest(unittest.TestCase):
    def load(self, items):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.json"
            path.write_text(json.dumps(items))
            return load_docs(path)

    def test_duplicate_id_is_named(self):
        with self.assertRaisesRegex(ValueError, "duplicate document id 'a'"):
            self.load([{"id": "a", "title": "x"}, {"id": "a", "title": "y"}])

    def test_missing_id_reports_position(self):
        with self.assertRaisesRegex(ValueError, "position 1"):
            self.load([{"id": "a"}, {"title": "no id"}])

    def test_id_is_not_a_field(self):
        self.assertEqual(self.load([{"id": "a", "title": "t"}]), {"a": {"title": "t"}})


if __name__ == "__main__":
    unittest.main()
