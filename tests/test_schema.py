"""Tests for reading CREATE TABLE schema from a real SQLite file (stdlib only)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from textsql.schema import get_schema


def _make_db(path: Path, statements: list[str]) -> None:
    con = sqlite3.connect(path)
    try:
        for s in statements:
            con.execute(s)
        con.commit()
    finally:
        con.close()


class TestGetSchema(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_create_statements_sorted_by_name(self):
        db = self.dir / "shop.sqlite"
        _make_db(db, [
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT)",
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        ])
        schema = get_schema(str(db))
        self.assertIn("CREATE TABLE customers", schema)
        self.assertIn("CREATE TABLE orders", schema)
        # Ordered by table name: customers before orders.
        self.assertLess(schema.index("customers"), schema.index("orders"))
        self.assertTrue(schema.rstrip().endswith(";"))

    def test_ignores_internal_sqlite_tables(self):
        db = self.dir / "auto.sqlite"
        # AUTOINCREMENT creates an internal sqlite_sequence table.
        _make_db(db, ["CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)"])
        schema = get_schema(str(db))
        self.assertNotIn("sqlite_sequence", schema)
        self.assertIn("CREATE TABLE t", schema)

    def test_raises_when_no_user_tables(self):
        db = self.dir / "empty.sqlite"
        # Force a valid db header, but leave no user tables.
        _make_db(db, ["CREATE TABLE _x (a)", "DROP TABLE _x"])
        with self.assertRaises(ValueError):
            get_schema(str(db))


if __name__ == "__main__":
    unittest.main()
