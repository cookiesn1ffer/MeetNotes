"""User corrections, keyterms, and few-shot example retrieval.

A single SQLite file holds three tables scoped by `team`:
    glossary               : (team, term, definition)         - STT keyterms + prompt glossary
    task_corrections       : (team, meeting_id, source_quote, correction_json)
    speaker_corrections    : (team, meeting_id, raw_label, name, clip_path)

Retrieval uses a tiny bag-of-words TF-IDF (no vector DB). Speaker corrections
contribute enrollment clips that `build_roster()` adds to the base roster;
`app.speakers.resolve` then averages the per-name clip embeddings on the next run.
"""

import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.speakers import Roster

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (
    team       TEXT NOT NULL,
    term       TEXT NOT NULL,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (team, term)
);
CREATE TABLE IF NOT EXISTS task_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    meeting_id      TEXT NOT NULL,
    source_quote    TEXT NOT NULL,
    correction_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speaker_corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    team       TEXT NOT NULL,
    meeting_id TEXT NOT NULL,
    raw_label  TEXT NOT NULL,
    name       TEXT NOT NULL,
    clip_path  TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tokenize(text: str) -> list[str]:
    # \w includes Devanagari under Python's default UNICODE flag.
    return re.findall(r"\w+", text.lower())


def _idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    n = len(docs_tokens) or 1
    # Smoothed IDF; +1 keeps unseen terms from dominating.
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    tf = Counter(tokens)
    length = len(tokens)
    return {t: (c / length) * idf.get(t, 0.0) for t, c in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class FeedbackStore:
    """Thin SQLite wrapper. Safe to construct per request; opens on demand."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------------- glossary

    def add_glossary_term(self, team: str, term: str, definition: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO glossary(team, term, definition, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(team, term) DO UPDATE SET definition=excluded.definition",
                (team, term, definition, _now()),
            )

    def glossary(self, team: str) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT term, definition FROM glossary WHERE team=? ORDER BY term",
                (team,),
            ).fetchall()
        return {r["term"]: r["definition"] for r in rows}

    def keyterms(self, team: str) -> list[str]:
        return list(self.glossary(team).keys())

    # ---------------------------------------------------------------- task corrections

    def record_task_correction(
        self, team: str, meeting_id: str, source_quote: str, correction: dict
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO task_corrections(team, meeting_id, source_quote, correction_json, "
                "created_at) VALUES (?,?,?,?,?)",
                (team, meeting_id, source_quote, json.dumps(correction, ensure_ascii=False), _now()),
            )
            return int(cur.lastrowid)

    def _all_task_corrections(self, team: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT source_quote, correction_json FROM task_corrections WHERE team=?",
                (team,),
            ).fetchall()
        return [
            {"source_quote": r["source_quote"], "correction": json.loads(r["correction_json"])}
            for r in rows
        ]

    def top_examples(self, team: str, transcript: str, k: int = 3) -> list[dict]:
        """Return up to `k` past corrections whose source_quote is most similar to `transcript`."""
        corrections = self._all_task_corrections(team)
        if not corrections or not transcript.strip():
            return []
        docs_tokens = [_tokenize(c["source_quote"]) for c in corrections]
        idf = _idf(docs_tokens)
        q = _tfidf_vec(_tokenize(transcript), idf)
        scored = sorted(
            zip(corrections, docs_tokens, strict=True),
            key=lambda pair: _cosine(q, _tfidf_vec(pair[1], idf)),
            reverse=True,
        )
        top = [c for c, _ in scored[:k]]
        # Drop zero-similarity hits (nothing lexically in common).
        top = [
            c for c, toks in zip(top, [t for _, t in scored[:k]], strict=True)
            if _cosine(q, _tfidf_vec(toks, idf)) > 0
        ]
        return top

    # ---------------------------------------------------------------- speaker corrections

    def record_speaker_correction(
        self,
        team: str,
        meeting_id: str,
        raw_label: str,
        name: str,
        clip_path: Path | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO speaker_corrections(team, meeting_id, raw_label, name, clip_path, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (team, meeting_id, raw_label, name, str(clip_path) if clip_path else None, _now()),
            )
            return int(cur.lastrowid)

    def enrollments_for(self, team: str, name: str) -> list[Path]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT clip_path FROM speaker_corrections "
                "WHERE team=? AND name=? AND clip_path IS NOT NULL "
                "ORDER BY id",
                (team, name),
            ).fetchall()
        return [Path(r["clip_path"]) for r in rows]

    def build_roster(self, team: str, base: dict[str, list[Path]]) -> Roster:
        """Merge `base` enrollments with every stored speaker-correction clip."""
        merged = {name: list(paths) for name, paths in base.items()}
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT name FROM speaker_corrections WHERE team=?", (team,)
            ).fetchall()
        for row in rows:
            name = row["name"]
            merged.setdefault(name, [])
            merged[name].extend(self.enrollments_for(team, name))
        return Roster({n: p for n, p in merged.items()})


def pipeline_inputs(store: FeedbackStore, team: str, transcript: str) -> dict:
    """Shortcut used by the pipeline before each new meeting.

    Returns {keyterms, glossary, examples} - keyterms go to Scribe,
    glossary + examples go into the extraction prompt.
    """
    t0 = time.perf_counter()
    result = {
        "keyterms": store.keyterms(team),
        "glossary": store.glossary(team),
        "examples": store.top_examples(team, transcript, k=3),
    }
    log.info(
        "stage=feedback_lookup elapsed_s=%.3f keyterms=%d examples=%d",
        time.perf_counter() - t0,
        len(result["keyterms"]),
        len(result["examples"]),
    )
    return result
