"""Resolve speakers on eval/meetings/, apply 3 simulated corrections, re-run, compare.

For each meeting under eval/meetings/<id>/ (same `data.json` layout as speaker_eval.py):
  1. Resolve speakers with an empty FeedbackStore. Record accuracy.
  2. Pick up to 3 ground-truth-labeled utterances across the eval set and apply
     `correct_speaker(team, raw_label, truth, audio=..., start=..., end=...)`.
     Each correction folds a fresh ECAPA embedding into the running mean per name.
  3. Resolve again using `store.build_roster()`. Record accuracy.
  4. Print before/after per meeting and overall.

PASS: overall accuracy after >= overall accuracy before.

Uses ECAPA (speechbrain, CPU) when the optional deps are installed; otherwise
reports that only the transcript heuristic is in play and still prints the two
numbers so the before/after comparison stays meaningful.

Usage: uv run python scripts/learning_demo.py
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.feedback import FeedbackStore
from app.models import Utterance
from app.speakers import Roster, ecapa_embedder, resolve

EVAL_DIR = ROOT / "eval" / "meetings"
TEAM = "demo"
MAX_CORRECTIONS = 3


def _load_meeting(d: Path):
    data = json.loads((d / "data.json").read_text(encoding="utf-8"))
    utterances = [
        Utterance(u["speaker"], u["start"], u["end"], u["text"], u.get("lang"))
        for u in data["utterances"]
    ]
    truth = [u["truth"] for u in data["utterances"]]
    audio = d / data["audio"] if data.get("audio") else None
    return utterances, truth, audio


def _accuracy(resolved: list[Utterance], truth: list[str]) -> float:
    if not resolved:
        return 0.0
    return sum(1 for r, t in zip(resolved, truth, strict=True) if r.speaker == t) / len(resolved)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not EVAL_DIR.is_dir():
        print(f"No eval directory at {EVAL_DIR}")
        return 0
    meetings = sorted(p for p in EVAL_DIR.iterdir() if p.is_dir())
    if not meetings:
        print(f"No meetings under {EVAL_DIR}")
        return 0

    try:
        embed_turn, _ = ecapa_embedder()
        voice_on = True
    except RuntimeError as exc:
        print(f"(ECAPA unavailable: {exc})")
        embed_turn = None
        voice_on = False

    # Collect all meetings' data up front.
    loaded = [(d, *_load_meeting(d)) for d in meetings]

    # ----- Before: fresh store, baseline roster has the names only (no embeddings).
    with tempfile.TemporaryDirectory() as td:
        store = FeedbackStore(Path(td) / "demo.db")
        # Seed the roster with the ground-truth name set so the transcript heuristic
        # has something to bind to; no embeddings yet.
        all_names = sorted({t for _, _, truth, _ in loaded for t in truth})
        base = {name: [] for name in all_names}

        totals_before = [0, 0]
        before_rows: list[tuple[str, int, int]] = []
        for d, utts, truth, audio in loaded:
            roster = Roster(base, {})
            resolved = resolve(utts, roster, audio=audio, embed_turn=embed_turn)
            correct = sum(1 for r, t in zip(resolved, truth, strict=True) if r.speaker == t)
            before_rows.append((d.name, correct, len(resolved)))
            totals_before[0] += correct
            totals_before[1] += len(resolved)

        # ----- Apply up to 3 simulated corrections across the eval set.
        corrections_made = 0
        if voice_on:
            for d, utts, truth, audio in loaded:
                if corrections_made >= MAX_CORRECTIONS or audio is None:
                    break
                for utt, t in zip(utts, truth, strict=True):
                    if corrections_made >= MAX_CORRECTIONS:
                        break
                    if utt.end - utt.start < 0.5:
                        continue
                    store.correct_speaker(
                        TEAM, utt.speaker, t,
                        audio=audio, start=utt.start, end=utt.end,
                        embedder=embed_turn,
                    )
                    corrections_made += 1
        print(f"Applied {corrections_made} simulated corrections.")

        # ----- After: re-resolve with the enriched roster.
        totals_after = [0, 0]
        after_rows: list[tuple[str, int, int]] = []
        for d, utts, truth, audio in loaded:
            roster = store.build_roster(TEAM, base)
            resolved = resolve(utts, roster, audio=audio, embed_turn=embed_turn)
            correct = sum(1 for r, t in zip(resolved, truth, strict=True) if r.speaker == t)
            after_rows.append((d.name, correct, len(resolved)))
            totals_after[0] += correct
            totals_after[1] += len(resolved)

    # ----- Report.
    print(f"\n{'meeting':30s}  {'before':>10s}  {'after':>10s}")
    for (name, cb, nb), (_, ca, na) in zip(before_rows, after_rows, strict=True):
        print(f"{name:30s}  {cb}/{nb} ({cb / nb:.1%})  {ca}/{na} ({ca / na:.1%})")
    before = totals_before[0] / totals_before[1] if totals_before[1] else 0.0
    after = totals_after[0] / totals_after[1] if totals_after[1] else 0.0
    print(f"\nOVERALL before: {before:.1%}   after: {after:.1%}")
    verdict = "PASS" if after >= before else "FAIL"
    print(f"{verdict}: after >= before (threshold-trivial comparison)")
    return 0 if after >= before else 1


if __name__ == "__main__":
    raise SystemExit(main())
