"""Run the speaker resolver over eval/meetings/<id>/ and report per-file accuracy.

Each meeting directory contains:
    data.json   with keys:
        audio      (optional): relative path to the audio file
        roster     : {name: [enrollment paths]}  (paths are relative to the dir)
        utterances : [{speaker, start, end, text, lang, truth}]
        threshold  (optional): cosine threshold (defaults to app.speakers.DEFAULT_THRESHOLD)

If audio and enrollments are present and speechbrain is installed, the embedding
pass runs; otherwise only the transcript pass runs.

PASS = overall accuracy >= 0.90. Per-file accuracy is reported either way.

Usage: uv run python scripts/speaker_eval.py
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import Utterance
from app.speakers import DEFAULT_THRESHOLD, Roster, ecapa_embedder, resolve

EVAL_DIR = ROOT / "eval" / "meetings"
PASS_THRESHOLD = 0.90


def _load(dir_: Path) -> tuple[list[Utterance], list[str], Roster, Path | None, float]:
    data = json.loads((dir_ / "data.json").read_text(encoding="utf-8"))
    utts = [
        Utterance(u["speaker"], u["start"], u["end"], u["text"], u.get("lang"))
        for u in data["utterances"]
    ]
    truth = [u["truth"] for u in data["utterances"]]
    roster = Roster(
        {name: [dir_ / p for p in paths] for name, paths in data.get("roster", {}).items()}
    )
    audio = dir_ / data["audio"] if data.get("audio") else None
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    return utts, truth, roster, audio, threshold


def _try_embedder():
    try:
        return ecapa_embedder()
    except RuntimeError as exc:
        print(f"(embedding pass disabled: {exc})")
        return None, None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not EVAL_DIR.is_dir():
        print(f"No eval directory at {EVAL_DIR}")
        return 1

    meetings = sorted(p for p in EVAL_DIR.iterdir() if p.is_dir())
    if not meetings:
        print(f"No meetings under {EVAL_DIR}")
        return 1

    embed_turn, embed_sample = _try_embedder()

    total_correct = total = 0
    for mdir in meetings:
        utts, truth, roster, audio, threshold = _load(mdir)
        resolved = resolve(
            utts, roster,
            audio=audio if embed_turn else None,
            threshold=threshold,
            embed_turn=embed_turn, embed_sample=embed_sample,
        )
        correct = sum(1 for r, t in zip(resolved, truth, strict=True) if r.speaker == t)
        n = len(resolved)
        total_correct += correct
        total += n
        print(f"{mdir.name:30s}  {correct}/{n}  ({correct / n:.1%})")

    overall = total_correct / total if total else 0.0
    print(f"\nOVERALL                       {total_correct}/{total}  ({overall:.1%})")
    verdict = "PASS" if overall >= PASS_THRESHOLD else "FAIL"
    print(f"{verdict}: threshold is {PASS_THRESHOLD:.0%}")
    return 0 if overall >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
