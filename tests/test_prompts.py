"""Tests for prompt construction and SQL extraction (stdlib only)."""
import unittest

from textsql.prompts import build_messages, build_user_prompt, extract_sql


class TestExtractSql(unittest.TestCase):
    def test_plain_query(self):
        self.assertEqual(extract_sql("SELECT a FROM t"), "SELECT a FROM t")

    def test_strips_markdown_fence(self):
        self.assertEqual(extract_sql("```sql\nSELECT a FROM t\n```"), "SELECT a FROM t")

    def test_strips_bare_fence(self):
        self.assertEqual(extract_sql("```\nSELECT 1\n```"), "SELECT 1")

    def test_strips_leading_label(self):
        self.assertEqual(extract_sql("SQL: SELECT a FROM t"), "SELECT a FROM t")
        self.assertEqual(extract_sql("query:  SELECT 1"), "SELECT 1")

    def test_first_statement_only(self):
        self.assertEqual(extract_sql("SELECT 1; SELECT 2"), "SELECT 1")

    def test_drops_trailing_prose(self):
        out = extract_sql("SELECT a FROM t;\nThis query returns all rows.")
        self.assertEqual(out, "SELECT a FROM t")

    def test_collapses_whitespace(self):
        self.assertEqual(extract_sql("SELECT a\n  FROM   t"), "SELECT a FROM t")

    def test_fence_with_label_inside(self):
        self.assertEqual(extract_sql("```sql\nSQL: SELECT 1\n```"), "SELECT 1")

    def test_empty_inputs(self):
        self.assertEqual(extract_sql(""), "")
        self.assertEqual(extract_sql(None), "")
        self.assertEqual(extract_sql("   \n  "), "")


class TestPromptBuilders(unittest.TestCase):
    def test_user_prompt_contains_schema_and_question(self):
        p = build_user_prompt("CREATE TABLE t (a INT);", "How many rows?")
        self.assertIn("CREATE TABLE t (a INT);", p)
        self.assertIn("How many rows?", p)
        self.assertTrue(p.rstrip().endswith("SQL:"))
        self.assertNotIn("Knowledge:", p)

    def test_user_prompt_includes_evidence_when_given(self):
        p = build_user_prompt("S", "Q", evidence="use column x")
        self.assertIn("Knowledge: use column x", p)

    def test_build_messages_shape(self):
        msgs = build_messages("S", "Q")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("Q", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
