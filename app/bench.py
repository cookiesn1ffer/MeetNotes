"""Benchmark the full pipeline against eval clips.

For each clip under eval/audio/ with matching truth under eval/truth/<id>.json,
run STT -> speaker resolver -> single LLM call, time each stage, score the
hypothesis against truth, and write bench/results.md as a markdown table.

Columns: clip, duration (s), time-to-notes (s), STT time (s), LLM time (s),
         WER (%), speaker acc (%), task precision (%), task recall (%)

PASS thresholds:
    time-to-notes <= 0.25 * duration (so < 15s for a 60s clip)
    speaker accuracy  >= 90%
    task recall       >= 80%

Truth file schema (eval/truth/<id>.json):
    {
      "duration_s":  float,            # length of the clip in seconds
      "transcript":  str,              # reference transcript (used for WER)
      "utterances":  [{"speaker": str, "start": float, "end": float, "text": str, "lang": str?}],
      "roster":      [str],            # candidate assignees + speaker names
      "tasks":       [{"assignee": str, "task": str, "deadline": str?}],
      "team_id":     str (optional, default "default")
    }

Usage: `uv run python -m app.bench`
"""

import argparse
import difflib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from app.feedback import FeedbackStore
from app.models import Utterance
from app.pipeline import run_pipeline

log = logging.getLogger(__name__)

EVAL_AUDIO = Path("eval") / "audio"
EVAL_TRUTH = Path("eval") / "truth"
BENCH_RESULTS = Path("bench") / "results.md"

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm"}

TASK_TEXT_MATCH_CUTOFF = 0.6  # SequenceMatcher ratio for fuzzy task-text match
TIME_TO_NOTES_RATIO = 0.25    # <= clip duration * this -> PASS
SPEAKER_PASS = 0.90
TASK_RECALL_PASS = 0.80

COLUMNS = [
    "clip",
    "duration (s)",
    "time-to-notes (s)",
    "STT time (s)",
    "LLM time (s)",
    "WER (%)",
    "speaker acc (%)",
    "task precision (%)",
    "task recall (%)",
]


# ---------------------------------------------------------------- metrics


def wer(reference: str, hypothesis: str) -> float:
    """Standard word error rate via Levenshtein distance over whitespace tokens."""
    r = reference.split()
    h = hypothesis.split()
    if not r:
        return 0.0 if not h else 1.0
    # Dynamic programming edit distance (O(len(r) * len(h))).
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(r)][len(h)] / len(r)


def speaker_accuracy(hyp: list[Utterance], truth: list[Utterance]) -> float:
    """Time-weighted speaker accuracy: for every truth span, fraction of its duration
    covered by a hypothesis span whose speaker name matches.
    """
    total = sum(u.end - u.start for u in truth)
    if total <= 0:
        return 0.0
    correct = 0.0
    for t in truth:
        for h in hyp:
            overlap = max(0.0, min(h.end, t.end) - max(h.start, t.start))
            if overlap > 0 and h.speaker == t.speaker:
                correct += overlap
    return correct / total


def _tasks_match(h: dict, t: dict) -> bool:
    if h.get("assignee", "") != t.get("assignee", ""):
        return False
    ratio = difflib.SequenceMatcher(
        None, (h.get("task") or "").lower(), (t.get("task") or "").lower()
    ).ratio()
    return ratio >= TASK_TEXT_MATCH_CUTOFF


def task_metrics(hyp: list[dict], truth: list[dict]) -> tuple[float, float]:
    """Greedy one-to-one matching; returns (precision, recall)."""
    matched: set[int] = set()
    tp = 0
    for h in hyp:
        for i, t in enumerate(truth):
            if i in matched:
                continue
            if _tasks_match(h, t):
                matched.add(i)
                tp += 1
                break
    # No predictions => no false positives (precision 1.0); no truth => no false
    # negatives (recall 1.0). This matches sklearn's zero_division=1 convention.
    precision = tp / len(hyp) if hyp else 1.0
    recall = tp / len(truth) if truth else 1.0
    return precision, recall


# ---------------------------------------------------------------- per-clip run


def _utterances_from_truth(truth: dict) -> list[Utterance]:
    return [
        Utterance(u["speaker"], u["start"], u["end"], u["text"], u.get("lang"))
        for u in truth["utterances"]
    ]


def _find_truth(audio: Path, truth_dir: Path) -> Path | None:
    candidate = truth_dir / f"{audio.stem}.json"
    return candidate if candidate.is_file() else None


def run_clip(
    audio: Path,
    truth: dict,
    *,
    store: FeedbackStore | None = None,
    post: Callable | None = None,
    call: Callable | None = None,
    use_cache: bool = True,
) -> dict:
    """Run the full pipeline on one clip (via `run_pipeline`); return a metric row."""
    team_id = truth.get("team_id", "default")
    roster_names = truth.get("roster", [])

    result = run_pipeline(
        audio, team_id, roster_names, store,
        post=post, call=call, use_cache=use_cache,
    )

    truth_utts = _utterances_from_truth(truth)
    truth_transcript = truth.get("transcript", " ".join(u.text for u in truth_utts))
    hyp_transcript = " ".join(u.text for u in result.utterances)

    wer_pct = wer(truth_transcript, hyp_transcript) * 100
    spk_pct = speaker_accuracy(result.utterances, truth_utts) * 100
    precision, recall = task_metrics(result.tasks, truth.get("tasks", []))

    return {
        "clip": audio.name,
        "duration (s)": float(truth.get("duration_s", 0.0)),
        "time-to-notes (s)": result.total_seconds,
        "STT time (s)": result.stage_seconds.get("stt", 0.0),
        "LLM time (s)": result.stage_seconds.get("extract", 0.0),
        "WER (%)": wer_pct,
        "speaker acc (%)": spk_pct,
        "task precision (%)": precision * 100,
        "task recall (%)": recall * 100,
    }


# ---------------------------------------------------------------- markdown rendering


def _fmt(col: str, value) -> str:
    if col == "clip":
        return str(value)
    if "(%)" in col:
        return f"{value:.1f}"
    return f"{value:.2f}"


def render_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(_fmt(col, r.get(col, "")) for col in COLUMNS) + " |"
        for r in rows
    ]
    verdict_lines = ["", "_PASS thresholds: time-to-notes <= 0.25 * duration, speaker acc >= 90%, task recall >= 80%._"]
    return "\n".join(["# Bench results", "", header, sep, *body, *verdict_lines]) + "\n"


# ---------------------------------------------------------------- CLI


def run_bench(
    audio_dir: Path = EVAL_AUDIO,
    truth_dir: Path = EVAL_TRUTH,
    out_path: Path = BENCH_RESULTS,
    *,
    store: FeedbackStore | None = None,
    post: Callable | None = None,
    call: Callable | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Walk `audio_dir`, pair with truth, run each clip, write a markdown table.

    Returns the list of metric rows (same shape as run_clip output).
    """
    rows: list[dict] = []
    if audio_dir.is_dir():
        for audio in sorted(audio_dir.iterdir()):
            if audio.suffix.lower() not in AUDIO_EXTS:
                continue
            truth_path = _find_truth(audio, truth_dir)
            if truth_path is None:
                log.warning("No truth file for %s; skipping", audio.name)
                continue
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            rows.append(
                run_clip(audio, truth, store=store, post=post, call=call, use_cache=use_cache)
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(rows), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the MeetNotes pipeline benchmark.")
    parser.add_argument("--audio-dir", type=Path, default=EVAL_AUDIO)
    parser.add_argument("--truth-dir", type=Path, default=EVAL_TRUTH)
    parser.add_argument("--out", type=Path, default=BENCH_RESULTS)
    args = parser.parse_args(argv)

    rows = run_bench(args.audio_dir, args.truth_dir, args.out)
    print(f"Wrote {args.out} ({len(rows)} clip(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
