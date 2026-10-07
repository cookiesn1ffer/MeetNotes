from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    speaker: str  # raw diarization label, e.g. "speaker_0"
    start: float  # seconds
    end: float  # seconds
    text: str  # as spoken; code-switching preserved
    lang: str | None  # ISO 639 code reported by Scribe, e.g. "hin", "eng"
