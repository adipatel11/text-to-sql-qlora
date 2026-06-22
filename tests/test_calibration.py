"""Tests for calibration sample construction (stdlib only, fake tokenizer)."""
import json
import tempfile
import unittest
from pathlib import Path

from textsql.export.calibration import build_calibration_texts


class FakeTokenizer:
    """Minimal stand-in: renders messages the way a chat template would."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        body = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
        return body + (" <gen>" if add_generation_prompt else "")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "data.jsonl"
        self.tok = FakeTokenizer()

    def tearDown(self):
        self._tmp.cleanup()

    def test_uses_stored_messages_and_appends_generation_prompt(self):
        _write_jsonl(self.path, [
            {"messages": [{"role": "system", "content": "S"},
                          {"role": "user", "content": "Q1"}]},
        ])
        texts = build_calibration_texts(self.path, self.tok, n_samples=1)
        self.assertEqual(len(texts), 1)
        self.assertIn("user:Q1", texts[0])
        self.assertTrue(texts[0].endswith("<gen>"))

    def test_falls_back_to_schema_and_question(self):
        _write_jsonl(self.path, [{"schema": "CREATE TABLE t (a)", "question": "Qx"}])
        texts = build_calibration_texts(self.path, self.tok, n_samples=1)
        self.assertIn("Qx", texts[0])
        self.assertIn("CREATE TABLE t (a)", texts[0])

    def test_caps_at_n_samples(self):
        _write_jsonl(self.path, [
            {"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(10)
        ])
        texts = build_calibration_texts(self.path, self.tok, n_samples=3)
        self.assertEqual(len(texts), 3)

    def test_sampling_is_deterministic_for_a_seed(self):
        _write_jsonl(self.path, [
            {"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(20)
        ])
        a = build_calibration_texts(self.path, self.tok, n_samples=5, seed=7)
        b = build_calibration_texts(self.path, self.tok, n_samples=5, seed=7)
        c = build_calibration_texts(self.path, self.tok, n_samples=5, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_empty_file_raises(self):
        _write_jsonl(self.path, [])
        with self.assertRaises(ValueError):
            build_calibration_texts(self.path, self.tok)


if __name__ == "__main__":
    unittest.main()
