"""Run every audio file in eval/audio/ through Scribe and print the transcript.

PASS for a file = the raw transcript contains both Devanagari (Hindi) and Latin (English)
text, i.e. neither language was dropped. Files that come back in a single script are
reported as CHECK so a human can judge them (all-Roman Hinglish is possible).

Usage: uv run python scripts/hinglish_check.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.stt import to_roman, transcribe

AUDIO_DIR = ROOT / "eval" / "audio"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm"}


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def has_latin(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in text)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files in {AUDIO_DIR}")
        return 1
    failed = 0
    for path in files:
        print(f"\n=== {path.name} ===")
        try:
            utterances = transcribe(path)
        except Exception as exc:  # noqa: BLE001 - report per file, keep going
            print(f"ERROR: {exc}")
            failed += 1
            continue
        for u in utterances:
            print(f"[{u.start:7.2f}-{u.end:7.2f}] {u.speaker} ({u.lang}): {u.text}")
            if has_devanagari(u.text):
                print(f"{'':>25}roman: {to_roman(u.text)}")
        raw = " ".join(u.text for u in utterances)
        if has_devanagari(raw) and has_latin(raw):
            print("PASS: both Hindi and English present")
        else:
            print("CHECK: single script only; verify by eye that no language was dropped")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
