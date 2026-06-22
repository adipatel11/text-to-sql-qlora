"""Execution accuracy: does the predicted query return the same rows as gold?

This is the honest metric for text-to-SQL. We run both queries against the
real SQLite database and compare result sets:

  * If the gold query has an ORDER BY, row order must match.
  * Otherwise rows are compared as a multiset (order-insensitive).
  * Column order is significant (same as the common Spider execution match).

Caveats vs. the official Spider *test-suite* evaluator: that harness runs each
query against many perturbed databases to catch queries that coincidentally
agree on one DB, and it canonicalizes column permutations. This single-DB
comparator is the standard, lighter "execution match" and is what most people
report; swap in the test-suite eval later for publication-grade rigor (noted
in the README).
"""
from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any


class QueryError(Exception):
    """Raised when a query fails to execute (syntax error, no such table...)."""


class QueryTimeout(Exception):
    """Raised when a query exceeds the time budget."""


def _connect(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    return con


def run_query(db_path: str, query: str, timeout: float = 30.0) -> list[tuple]:
    """Execute `query` read-only with a wall-clock timeout.

    sqlite's Connection.interrupt() is safe to call from another thread, so we
    run the query in a worker thread and interrupt it if it overruns.
    """
    con = _connect(db_path)
    out: dict[str, Any] = {}

    def _work() -> None:
        try:
            cur = con.cursor()
            cur.execute(query)
            out["rows"] = cur.fetchall()
        except Exception as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        con.interrupt()
        t.join(1.0)
        con.close()
        raise QueryTimeout(f"query exceeded {timeout}s")
    con.close()
    if "error" in out:
        raise QueryError(str(out["error"]))
    return out["rows"]


def _norm_cell(c: Any) -> str:
    # Canonicalize each cell to a string so values are hashable for the
    # multiset comparison and so 1, 1.0, and "1" all compare equal. SQL freely
    # returns an int or a float for the same logical value (COUNT vs AVG, SUM
    # over a join, ...), so an integral float must fold onto its int form or we
    # report false mismatches and understate execution accuracy.
    if c is None:
        return ""
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c)


def _normalize(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_cell(c) for c in row) for row in rows]


def results_match(gold_rows: list[tuple], pred_rows: list[tuple], order_matters: bool) -> bool:
    g, p = _normalize(gold_rows), _normalize(pred_rows)
    if order_matters:
        return g == p
    return Counter(g) == Counter(p)


def execution_match(
    db_path: str,
    gold_sql: str,
    pred_sql: str,
    timeout: float = 30.0,
) -> tuple[bool, str]:
    """Return (is_correct, status).

    status is one of: "match", "mismatch", "pred_error", "pred_timeout",
    "gold_error" (the last means the reference itself failed -- a data problem,
    not a model error).
    """
    try:
        gold_rows = run_query(db_path, gold_sql, timeout)
    except (QueryError, QueryTimeout) as e:
        return False, f"gold_error: {e}"

    if not pred_sql.strip():
        return False, "pred_error: empty"
    try:
        pred_rows = run_query(db_path, pred_sql, timeout)
    except QueryTimeout:
        return False, "pred_timeout"
    except QueryError as e:
        return False, f"pred_error: {e}"

    order_matters = "order by" in gold_sql.lower()
    ok = results_match(gold_rows, pred_rows, order_matters)
    return ok, "match" if ok else "mismatch"
