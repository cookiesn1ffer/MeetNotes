"""Shape a MeetingResult into the JSON the web UI consumes.

The UI groups work per person (cards with notes + tasks) and shows a transcript with a
language tag and mm:ss timestamp per line. The pipeline produces flat tasks, notes grouped
by topic, and utterances; this module is the one place that maps between the two.
"""

import re

from app.extract import _match_assignee
from app.pipeline import MeetingResult

GENERAL_CARD = "Meeting"  # holds the summary + notes not tied to one person
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_LATIN = re.compile("[A-Za-z]")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "person"


def mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def line_language(text: str, lang: str | None) -> str:
    """'HI + EN' for code-switched lines (both scripts), else from Scribe's language code."""
    has_hi, has_en = bool(_DEVANAGARI.search(text)), bool(_LATIN.search(text))
    if has_hi and has_en:
        return "HI + EN"
    if has_hi or lang in ("hin", "hi"):
        return "HI"
    return "EN"


def speaker_confidence(result: MeetingResult) -> float | None:
    """Percent of speaking time attributed to a named speaker (not 'Unknown N')."""
    total = sum(max(0.0, u.end - u.start) for u in result.utterances)
    if total <= 0:
        return None
    named = sum(
        max(0.0, u.end - u.start)
        for u in result.utterances
        if not u.speaker.startswith("Unknown ")
    )
    return round(100 * named / total, 1)


def _note_text(notes: list[dict]) -> str:
    lines: list[str] = []
    for note in notes:
        topic = note.get("topic", "")
        for point in note.get("points", []):
            lines.append(f"{topic}: {point}" if topic else point)
    return "\n".join(lines)


def tasks_with_ids(tasks: list[dict]) -> list[dict]:
    return [{"id": f"task-{i + 1}", **t} for i, t in enumerate(tasks)]


def build_people(result: MeetingResult) -> list[dict]:
    """One entry per roster name (in roster order) + a 'Meeting' card for general notes."""
    roster = list(result.roster)
    tasks = tasks_with_ids(result.tasks)

    notes_by_person: dict[str, list[dict]] = {name: [] for name in roster}
    general: list[dict] = []
    for note in result.notes:
        owner = _match_assignee(note.get("person") or "", roster)
        (notes_by_person[owner] if owner else general).append(note)

    people = [
        {
            "id": _slug(name),
            "name": name,
            "notes": _note_text(notes_by_person[name]),
            "tasks": [t for t in tasks if t["assignee"] == name],
        }
        for name in roster
    ]
    general_text = "\n".join(p for p in (result.summary, _note_text(general)) if p)
    if general_text:
        people.append({"id": _slug(GENERAL_CARD), "name": GENERAL_CARD, "notes": general_text, "tasks": []})
    return people


def build_transcript(result: MeetingResult) -> list[dict]:
    return [
        {
            "id": f"line-{i + 1}",
            "speaker": u.speaker,
            "start": u.start,
            "end": u.end,
            "timestamp": mmss(u.start),
            "text": u.text,
            "lang": u.lang,
            "language": line_language(u.text, u.lang),
        }
        for i, u in enumerate(result.utterances)
    ]


def stage_timings(result: MeetingResult) -> dict[str, float]:
    s = result.stage_seconds
    return {
        "transcribe": round(s.get("stt", 0.0), 2),
        "speakers": round(s.get("speakers", 0.0), 2),
        "tasks": round(s.get("extract", 0.0), 2),
        "total": round(result.total_seconds, 2),
    }
