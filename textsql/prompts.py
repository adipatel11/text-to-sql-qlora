"""Prompt construction and SQL extraction.

Keep this module the single source of truth for prompt format. Training and
all eval/serving backends import from here so the prompt the model was tuned
on is byte-for-byte the prompt it is evaluated on.
"""
from __future__ import annotations

import re
from typing import Optional

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. "
    "Given a database schema and a natural-language question, respond with a "
    "single valid SQLite query that answers the question. "
    "Return only the SQL query: no explanation, no markdown code fences."
)


def build_user_prompt(
    schema: str,
    question: str,
    evidence: Optional[str] = None,
) -> str:
    """Render the user turn. `evidence` is the external-knowledge hint used by
    BIRD; it is ignored (None) for Spider."""
    parts = ["Database schema:", schema.strip(), ""]
    if evidence:
        parts += [f"Knowledge: {evidence.strip()}", ""]
    parts += [f"Question: {question.strip()}", "", "SQL:"]
    return "\n".join(parts)


def build_messages(
    schema: str,
    question: str,
    evidence: Optional[str] = None,
) -> list[dict]:
    """Chat-format messages. Backends call tokenizer.apply_chat_template on
    this (or post it to an OpenAI-compatible /chat/completions endpoint)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(schema, question, evidence)},
    ]


_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(?:sql|sqlite|query)\s*:\s*", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model completion.

    Handles markdown fences, a leading 'SQL:' label, and trailing prose, and
    returns the first statement (up to the first semicolon) on one line.
    """
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    text = _LABEL_RE.sub("", text)
    # First statement only.
    if ";" in text:
        text = text.split(";", 1)[0]
    # Collapse newlines/runs of whitespace so logs and comparisons are stable.
    text = " ".join(text.split())
    return text.strip()
