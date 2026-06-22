"""Tests for execution-accuracy comparison (stdlib only)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from textsql.eval.execution import (
    QueryError,
    QueryTimeout,
    execution_match,
    results_match,
    run_query,
)


class TestResultsMatch(unittest.TestCase):
    def test_order_insensitive_multiset(self):
        a = [("1", "x"), ("2", "y")]
        b = [("2", "y"), ("1", "x")]
        self.assertTrue(results_match(a, b, order_matters=False))
        self.assertFalse(results_match(a, b, order_matters=True))

    def test_type_coercion(self):
        # 1 == 1.0 == "1" should not silently mismatch (int vs float vs str).
        self.assertTrue(results_match([(1,)], [(1.0,)], order_matters=False))
        self.assertTrue(results_match([(1,)], [("1",)], order_matters=False))
        # But a genuine fractional difference must still mismatch.
        self.assertFalse(results_match([(3,)], [(3.5,)], order_matters=False))
        self.assertTrue(results_match([(3.5,)], [(3.5,)], order_matters=False))

    def test_none_treated_as_empty_string(self):
        self.assertTrue(results_match([(None,)], [("",)], order_matters=False))

    def test_duplicates_are_significant(self):
        self.assertFalse(results_match([(1,), (1,)], [(1,)], order_matters=False))


class TestExecution(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "t.sqlite")
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE nums (n INTEGER)")
        con.executemany("INSERT INTO nums (n) VALUES (?)", [(1,), (2,), (3,)])
        con.commit()
        con.close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_query_readonly_blocks_writes(self):
        with self.assertRaises(QueryError):
            run_query(self.db, "INSERT INTO nums (n) VALUES (4)")

    def test_match(self):
        ok, status = execution_match(
            self.db, "SELECT n FROM nums", "SELECT n FROM nums WHERE n IN (1,2,3)"
        )
        self.assertTrue(ok)
        self.assertEqual(status, "match")

    def test_mismatch(self):
        ok, status = execution_match(
            self.db, "SELECT n FROM nums", "SELECT n FROM nums WHERE n = 1"
        )
        self.assertFalse(ok)
        self.assertEqual(status, "mismatch")

    def test_order_by_is_order_sensitive(self):
        gold = "SELECT n FROM nums ORDER BY n ASC"
        ok, _ = execution_match(self.db, gold, "SELECT n FROM nums ORDER BY n DESC")
        self.assertFalse(ok)
        ok, _ = execution_match(self.db, gold, "SELECT n FROM nums ORDER BY n ASC")
        self.assertTrue(ok)

    def test_no_order_by_ignores_order(self):
        ok, _ = execution_match(
            self.db, "SELECT n FROM nums", "SELECT n FROM nums ORDER BY n DESC"
        )
        self.assertTrue(ok)

    def test_pred_error(self):
        ok, status = execution_match(self.db, "SELECT n FROM nums", "SELECT bad syntax (")
        self.assertFalse(ok)
        self.assertTrue(status.startswith("pred_error"))

    def test_empty_pred(self):
        ok, status = execution_match(self.db, "SELECT n FROM nums", "   ")
        self.assertFalse(ok)
        self.assertEqual(status, "pred_error: empty")

    def test_gold_error(self):
        ok, status = execution_match(self.db, "SELECT * FROM no_such_table", "SELECT 1")
        self.assertFalse(ok)
        self.assertTrue(status.startswith("gold_error"))

    def test_pred_timeout(self):
        # An effectively unbounded recursive CTE; interrupted by the time budget.
        slow = (
            "WITH RECURSIVE c(x) AS ("
            "  SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 1000000000"
            ") SELECT count(*) FROM c"
        )
        with self.assertRaises(QueryTimeout):
            run_query(self.db, slow, timeout=0.2)
        ok, status = execution_match(self.db, "SELECT n FROM nums", slow, timeout=0.2)
        self.assertFalse(ok)
        self.assertEqual(status, "pred_timeout")


if __name__ == "__main__":
    unittest.main()
