"""Full-pipeline orchestration: audio -> STT -> speaker resolve -> single LLM call.

Single place that wires the three pipeline stages:
    1. stt.transcribe       with glossary terms as Scribe keyterms
    2. speakers.resolve     with the team roster + stored voice embeddings
    3. extract.extract      with glossary + top-3 TF-IDF few-shot examples

Captures per-stage timing (both wall-clock seconds and the modules' own `stage=…`
log lines) and returns a MeetingResult. Used by:
    - app/api/main.py:/process     (web UI)
    - app/bench.py:run_clip        (benchmark harness)
    - app/cli.py:process           (`uv run python -m app process ...`)

Behavior-identical with the previous ad-hoc wiring in /process and bench — the
only observable addition is one extra `stage=feedback_lookup …` log line in the
captured log for the extraction prompt's example lookup.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.extract import extract
from app.feedback import FeedbackStore, pipeline_inputs
from app.models import Utterance
from app.speakers import Roster, resolve
from app.stt import transcribe

log = logging.getLogger(__name__)

DEFAULT_DB = Path("data/meetnotes.db")


@dataclass(frozen=True)
class MeetingResult:
    audio_path: Path
    team_id: str
    roster: list[str]
    utterances: list[Utterance]           # resolved utterances (speaker names mapped)
    summary: str
    notes: list[dict]                     # [{"topic": str, "points": [str]}]
    tasks: list[dict]                     # extract.py's post-processed task dicts
    stage_seconds: dict[str, float]       # {"stt": …, "speakers": …, "extract": …}
    stage_logs: list[str]                 # captured `stage=…` log lines
    total_seconds: float

    def to_dict(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "team_id": self.team_id,
            "roster": list(self.roster),
            "utterances": [
                {
                    "speaker": u.speaker, "start": u.start, "end": u.end,
                    "text": u.text, "lang": u.lang,
                }
                for u in self.utterances
            ],
            "summary": self.summary,
            "notes": self.notes,
            "tasks": self.tasks,
            "stage_seconds": dict(self.stage_seconds),
            "stage_logs": list(self.stage_logs),
            "total_seconds": self.total_seconds,
        }


class _StageLogCollector(logging.Handler):
    """Captures `stage=…` log lines from the pipeline modules during a request."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "stage=" in msg:
            self.lines.append(msg)


def run_pipeline(
    audio_path: Path,
    team_id: str,
    roster: list[str],
    store: FeedbackStore | None,
    *,
    post: Callable | None = None,
    call: Callable | None = None,
    use_cache: bool = True,
) -> MeetingResult:
    """Run STT -> speakers -> extract on `audio_path` and return a MeetingResult.

    `store` may be None (bench's test path), in which case no glossary / examples
    / stored embeddings are mixed in. `post` and `call` are injection points for
    tests to stay offline.
    """
    audio_path = Path(audio_path)
    roster = list(roster)

    collector = _StageLogCollector()
    app_log = logging.getLogger("app")
    prior_level = app_log.level
    app_log.setLevel(logging.INFO)
    app_log.addHandler(collector)

    stage_seconds: dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        # Phase 1: glossary -> Scribe keyterms. Direct call so no feedback_lookup
        # log line precedes STT; pipeline_inputs runs post-STT with the real transcript.
        keyterms = store.keyterms(team_id) if store is not None else []

        t0 = time.perf_counter()
        stt_kwargs = {"keyterms": keyterms, "use_cache": use_cache}
        if post is not None:
            stt_kwargs["post"] = post
        utterances = transcribe(audio_path, **stt_kwargs)
        stage_seconds["stt"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        base_roster = (
            store.build_roster(team_id, {name: [] for name in roster})
            if store is not None
            else Roster({name: [] for name in roster})
        )
        resolved = resolve(utterances, base_roster)
        stage_seconds["speakers"] = time.perf_counter() - t0

        # Phase 2: glossary + top-3 few-shot examples for extract. One pipeline_inputs
        # call captures both (keyterms are redundant here; glossary + examples carry).
        if store is not None:
            transcript = " ".join(u.text for u in resolved)
            inputs = pipeline_inputs(store, team_id, transcript)
            glossary = inputs["glossary"]
            examples = inputs["examples"]
        else:
            glossary, examples = {}, []

        t0 = time.perf_counter()
        extract_kwargs = {"glossary": glossary, "examples": examples}
        if call is not None:
            extract_kwargs["call"] = call
        extracted = extract(resolved, roster, **extract_kwargs)
        stage_seconds["extract"] = time.perf_counter() - t0
    finally:
        app_log.removeHandler(collector)
        app_log.setLevel(prior_level)

    total = time.perf_counter() - t_total
    log.info(
        "stage=pipeline elapsed_s=%.3f stt_s=%.3f speakers_s=%.3f extract_s=%.3f",
        total,
        stage_seconds.get("stt", 0.0),
        stage_seconds.get("speakers", 0.0),
        stage_seconds.get("extract", 0.0),
    )
    return MeetingResult(
        audio_path=audio_path,
        team_id=team_id,
        roster=roster,
        utterances=resolved,
        summary=extracted.get("summary", ""),
        notes=extracted.get("notes", []),
        tasks=extracted.get("tasks", []),
        stage_seconds=stage_seconds,
        stage_logs=collector.lines,
        total_seconds=total,
    )
