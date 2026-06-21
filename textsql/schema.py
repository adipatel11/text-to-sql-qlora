"""Extract a readable schema string from a SQLite database file.

Reading the real CREATE TABLE statements straight from the .sqlite file keeps
the schema faithful (types, primary/foreign keys) and avoids depending on
Spider's tables.json representation.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path


def _connect(db_path: str | Path) -> sqlite3.Connection:
    # Open read-only via URI so eval can never mutate the database.
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    return con


@lru_cache(maxsize=512)
def get_schema(db_path: str) -> str:
    """Return the concatenated CREATE TABLE statements for `db_path`."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        stmts = [row[0].strip().rstrip(";") for row in cur.fetchall()]
    finally:
        con.close()
    if not stmts:
        raise ValueError(f"No tables found in {db_path}")
    return ";\n\n".join(stmts) + ";"
