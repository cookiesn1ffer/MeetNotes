"""User corrections, team glossary, and few-shot example retrieval.

Two SQLite tables scoped by `team_id`:
    corrections(id, team_id, kind, before, after, context, created_at)
        kind = 'task'    -> before/after are JSON task dicts, context is the source_quote
        kind = 'speaker' -> before is the raw diarization label (e.g. 'speaker_0'),
                            after is the corrected name, context is the enrollment clip path
    glossary(team_id, term, kind, created_at)
        kind labels the term's category (person, product, acronym, ...).

Example retrieval uses scikit-learn's `TfidfVectorizer` + `cosine_similarity` over the
`context` (source_quote) of past task corrections. No vector DB. Speaker corrections
contribute enrollment clips that `build_roster()` adds to the base roster;
`app.speakers.resolve` averages the per-name clip embeddings on the next run.
"""

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.speakers import Embedder, Roster, ecapa_embedder

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    "before"   TEXT NOT NULL,
    "after"    TEXT NOT NULL,
    context    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_team_kind
    ON corrections(team_id, kind);
CREATE TABLE IF NOT EXISTS glossary (
    team_id    TEXT NOT NULL,
    term       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (team_id, term)
);
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    team_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    count      INTEGER NOT NULL,
    embedding  TEXT NOT NULL,       -- JSON array of floats (running mean)
    updated_at TEXT NOT NULL,
    PRIMARY KEY (team_id, name)
);
"""


_default_ecapa_embedder: Embedder | None = None


def _load_default_embedder() -> Embedder:
    """Lazy-load the ECAPA embedder on first use. Raises if speechbrain is missing."""
    global _default_ecapa_embedder
    if _default_ecapa_embedder is None:
        turn, _ = ecapa_embedder()
        _default_ecapa_embedder = turn
    return _default_ecapa_embedder


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Matches single-char and multi-char Unicode word tokens, so Devanagari and short
# Hinglish words both participate in TF-IDF. The sklearn default (\b\w\w+\b) drops
# every 1-char token.
_TOKEN_PATTERN = r"(?u)\b\w+\b"


# ---------------------------------------------------------------- store


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

    # -------- glossary

    def add_glossary_term(self, team_id: str, term: str, kind: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO glossary(team_id, term, kind, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(team_id, term) DO UPDATE SET kind=excluded.kind",
                (team_id, term, kind, _now()),
            )

    def glossary(self, team_id: str) -> dict[str, str]:
        """Return {term: kind} for the given team, ordered by term."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT term, kind FROM glossary WHERE team_id=? ORDER BY term",
                (team_id,),
            ).fetchall()
        return {r["term"]: r["kind"] for r in rows}

    def keyterms(self, team_id: str) -> list[str]:
        return list(self.glossary(team_id).keys())

    # -------- corrections (low-level)

    def _insert_correction(
        self, team_id: str, kind: str, before: str, after: str, context: str
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                'INSERT INTO corrections(team_id, kind, "before", "after", context, created_at) '
                "VALUES (?,?,?,?,?,?)",
                (team_id, kind, before, after, context, _now()),
            )
            return int(cur.lastrowid)

    # -------- task corrections

    def correct_task(
        self, team_id: str, before: dict, after: dict, source_quote: str
    ) -> int:
        return self._insert_correction(
            team_id,
            "task",
            json.dumps(before, ensure_ascii=False),
            json.dumps(after, ensure_ascii=False),
            source_quote,
        )

    def retrieve_examples(self, team_id: str, transcript: str, k: int = 3) -> list[dict]:
        """Return up to `k` past task corrections most similar to `transcript`.

        TF-IDF via `sklearn.feature_extraction.text.TfidfVectorizer` over the stored
        `source_quote` of each task correction; `sklearn.metrics.pairwise.cosine_similarity`
        ranks them against the current transcript. Zero-similarity rows are dropped.

        Each example: {"before": dict, "after": dict, "source_quote": str}.
        """
        with self._conn() as c:
            rows = c.execute(
                'SELECT "before", "after", context FROM corrections '
                "WHERE team_id=? AND kind=?",
                (team_id, "task"),
            ).fetchall()
        if not rows or not transcript.strip():
            return []
        docs = [r["context"] for r in rows]
        vec = TfidfVectorizer(token_pattern=_TOKEN_PATTERN, lowercase=True)
        try:
            matrix = vec.fit_transform([*docs, transcript])
        except ValueError:  # empty vocabulary (e.g. transcript shares no \w tokens)
            return []
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(zip(sims, rows, strict=True), key=lambda p: p[0], reverse=True)
        return [
            {
                "before": json.loads(r["before"]),
                "after": json.loads(r["after"]),
                "source_quote": r["context"],
            }
            for score, r in ranked[:k]
            if score > 0
        ]

    # -------- speaker corrections

    def correct_speaker(
        self,
        team_id: str,
        before_label: str,
        after_name: str,
        clip_path: Path | None = None,
        *,
        embedding: list[float] | None = None,
        audio: Path | None = None,
        start: float | None = None,
        end: float | None = None,
        embedder: Embedder | None = None,
    ) -> int:
        """Record a speaker correction and update that name's running-mean embedding.

        Supplying `embedding` uses it directly. Otherwise, if `audio/start/end` are
        given, the embedding is computed via `embedder` (defaults to lazy-loaded
        ECAPA on CPU). `clip_path`, if given, is persisted in the audit row only.
        A correction with no embedding source still records the audit row.
        """
        row_id = self._insert_correction(
            team_id,
            "speaker",
            before_label,
            after_name,
            str(clip_path) if clip_path else "",
        )
        vec: list[float] | None = embedding
        if vec is None and audio is not None and start is not None and end is not None:
            emb = embedder or _load_default_embedder()
            vec = emb(Path(audio), start, end)
        if vec is not None:
            self._update_speaker_mean(team_id, after_name, vec)
        return row_id

    def embed_span(
        self, audio: Path, start: float, end: float, embedder: Embedder | None = None
    ) -> list[float] | None:
        """Embed an audio span, or return None (with a logged reason) if that isn't possible.

        Embedding needs the optional speechbrain/torchaudio deps and a decodable file, so
        callers that must still record a correction use this instead of raising.
        """
        try:
            return (embedder or _load_default_embedder())(Path(audio), start, end)
        except Exception as exc:  # noqa: BLE001 - any embedder/decoder failure just means "no voice"
            log.warning("voice embedding skipped: %s", exc)
            return None

    def _update_speaker_mean(self, team_id: str, name: str, v: list[float]) -> None:
        """Fold `v` into the running mean for (team_id, name)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT count, embedding FROM speaker_embeddings WHERE team_id=? AND name=?",
                (team_id, name),
            ).fetchone()
            if row is None or len(json.loads(row["embedding"])) != len(v):
                # First sample for this name, or a dimension change (embedder swapped):
                # restart the mean from this vector.
                new_count = 1
                new_mean = list(v)
            else:
                n = row["count"]
                prev = json.loads(row["embedding"])
                new_count = n + 1
                new_mean = [
                    m + (x - m) / new_count for m, x in zip(prev, v, strict=True)
                ]
            c.execute(
                "INSERT INTO speaker_embeddings(team_id, name, count, embedding, updated_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(team_id, name) DO UPDATE SET "
                "count=excluded.count, embedding=excluded.embedding, updated_at=excluded.updated_at",
                (team_id, name, new_count, json.dumps(new_mean), _now()),
            )

    def get_speaker_embedding(self, team_id: str, name: str) -> list[float] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT embedding FROM speaker_embeddings WHERE team_id=? AND name=?",
                (team_id, name),
            ).fetchone()
        return json.loads(row["embedding"]) if row else None

    def speaker_embeddings(self, team_id: str) -> dict[str, list[float]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, embedding FROM speaker_embeddings WHERE team_id=?",
                (team_id,),
            ).fetchall()
        return {r["name"]: json.loads(r["embedding"]) for r in rows}

    def enrollments_for(self, team_id: str, name: str) -> list[Path]:
        """Clip paths recorded for `name` in the corrections audit table."""
        with self._conn() as c:
            rows = c.execute(
                'SELECT context FROM corrections '
                'WHERE team_id=? AND kind=? AND "after"=? AND context!="" '
                "ORDER BY id",
                (team_id, "speaker", name),
            ).fetchall()
        return [Path(r["context"]) for r in rows]

    def build_roster(self, team_id: str, base: dict[str, list[Path]]) -> Roster:
        """Merge `base` enrollments with every stored speaker correction.

        Returns a Roster carrying both enrollment file paths (clip-based, for the
        resolver's path-embedding fallback) and precomputed running-mean embeddings
        (preferred when present).
        """
        merged_paths = {name: list(paths) for name, paths in base.items()}
        with self._conn() as c:
            correction_rows = c.execute(
                'SELECT DISTINCT "after" FROM corrections WHERE team_id=? AND kind=?',
                (team_id, "speaker"),
            ).fetchall()
        for row in correction_rows:
            name = row["after"]
            merged_paths.setdefault(name, [])
            merged_paths[name].extend(self.enrollments_for(team_id, name))
        return Roster(merged_paths, self.speaker_embeddings(team_id))


def pipeline_inputs(store: FeedbackStore, team_id: str, transcript: str) -> dict:
    """Convenience: fetch everything the pipeline needs before a new meeting."""
    t0 = time.perf_counter()
    result = {
        "keyterms": store.keyterms(team_id),
        "glossary": store.glossary(team_id),
        "examples": store.retrieve_examples(team_id, transcript, k=3),
    }
    log.info(
        "stage=feedback_lookup elapsed_s=%.3f keyterms=%d examples=%d",
        time.perf_counter() - t0,
        len(result["keyterms"]),
        len(result["examples"]),
    )
    return result
