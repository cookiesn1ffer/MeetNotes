"""Single-LLM-call extraction of summary, notes, and tasks from a resolved transcript.

One call to Claude Messages API with tool-use forcing JSON output that matches SCHEMA.
The HTTP call is injectable so tests stay offline. Post-processing enforces the rules
the prompt states, as defense in depth:
  * tasks without a `source_quote` that appears in the transcript are dropped
  * `assignee` is fuzzy-matched to the roster (handles Kaushal/Koshal, कौशल, etc.)
"""

import difflib
import json
import logging
import time
import urllib.request
from collections.abc import Callable

from app.config import get_anthropic_key
from app.models import Utterance
from app.stt import to_roman

log = logging.getLogger(__name__)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5-5"
NAME_MATCH_CUTOFF = 0.7  # difflib ratio; "koshal" vs "kaushal" is ~0.77

LLMCall = Callable[[dict], dict]  # (request body) -> response body


SCHEMA = {
    "type": "object",
    "required": ["summary", "notes", "tasks"],
    "properties": {
        "summary": {"type": "string"},
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic", "points"],
                "properties": {
                    "topic": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "assignee",
                    "assigned_by",
                    "task",
                    "source_quote",
                    "confidence",
                ],
                "properties": {
                    "assignee": {"type": "string"},
                    "assigned_by": {"type": "string"},
                    "task": {"type": "string"},
                    "deadline": {"type": ["string", "null"]},
                    "source_quote": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


SYSTEM_RULES = """You extract summary + notes + tasks from a code-switching Hindi/English meeting transcript.

Hard rules:
- Call the tool `emit_meeting_notes` exactly once with the structured output.
- Preserve the speakers' Hinglish code-switching; do not translate.
- For every task:
  * `assigned_by` is the speaker who gave the command (the utterance's speaker).
  * `assignee` MUST be exactly one of the roster names provided. Match transliteration
    variants (Kaushal / Koshal / कौशल are the same person). If the command does
    not clearly target a roster name, DO NOT emit the task.
  * `source_quote` is a VERBATIM substring from the transcript containing the command.
  * `task` is a short description of the deliverable (e.g. "ppt on planets").
  * `deadline` is a short phrase as stated ("Friday", "next meeting", "कल तक"),
    or null when none is stated. Never invent dates.
  * `confidence` is in [0, 1]; use low values when the command is ambiguous.
- Ignore chit-chat, factual asides, questions, and anything that isn't a command/assignment.
- Group notes by topic. Keep topic names short.
"""


def _render_transcript(utterances: list[Utterance]) -> str:
    return "\n".join(f"[{u.start:7.2f}] {u.speaker}: {u.text}" for u in utterances)


def _prompt(
    utterances: list[Utterance],
    roster: list[str],
    glossary: dict[str, str],
    examples: list[dict],
) -> str:
    lines: list[str] = ["Roster (assignees must be from this list):"]
    lines.append("  " + (", ".join(roster) if roster else "(none)"))
    lines += ["", "Team glossary:"]
    if glossary:
        lines += [f"  {k}: {v}" for k, v in glossary.items()]
    else:
        lines.append("  (none)")
    lines += ["", "Correction examples (prior tasks and their correct extraction):"]
    if examples:
        lines += ["  " + json.dumps(ex, ensure_ascii=False) for ex in examples[:3]]
    else:
        lines.append("  (none)")
    lines += ["", "Transcript:", _render_transcript(utterances)]
    return "\n".join(lines)


def _call_claude(body: dict) -> dict:
    req = urllib.request.Request(
        MESSAGES_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": get_anthropic_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _match_assignee(name: str, roster: list[str]) -> str | None:
    """Return the roster entry closest to `name`, or None if nothing is close enough."""
    if not roster or not name:
        return None
    norm = {to_roman(n).lower(): n for n in roster}
    target = to_roman(name).lower()
    if target in norm:
        return norm[target]
    best_key = max(norm, key=lambda k: difflib.SequenceMatcher(None, target, k).ratio())
    ratio = difflib.SequenceMatcher(None, target, best_key).ratio()
    return norm[best_key] if ratio >= NAME_MATCH_CUTOFF else None


def _enforce_rules(payload: dict, utterances: list[Utterance], roster: list[str]) -> dict:
    transcript = " ".join(u.text for u in utterances)
    tasks: list[dict] = []
    for t in payload.get("tasks", []):
        quote = (t.get("source_quote") or "").strip()
        if not quote or quote not in transcript:
            continue
        resolved = _match_assignee(t.get("assignee", ""), roster)
        if not resolved:
            continue
        tasks.append(
            {
                "assignee": resolved,
                "assigned_by": t.get("assigned_by") or "",
                "task": t.get("task") or "",
                "deadline": t.get("deadline"),
                "source_quote": quote,
                "confidence": float(t.get("confidence", 0.0)),
            }
        )
    return {
        "summary": payload.get("summary", ""),
        "notes": payload.get("notes", []),
        "tasks": tasks,
    }


def extract(
    utterances: list[Utterance],
    roster: list[str],
    glossary: dict[str, str] | None = None,
    examples: list[dict] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    call: LLMCall | None = None,
) -> dict:
    """One LLM call. Returns {"summary", "notes", "tasks"} after rule enforcement."""
    t0 = time.perf_counter()
    body = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_RULES,
        "tools": [
            {
                "name": "emit_meeting_notes",
                "description": "Emit the structured meeting output.",
                "input_schema": SCHEMA,
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_meeting_notes"},
        "messages": [
            {
                "role": "user",
                "content": _prompt(utterances, roster, glossary or {}, examples or []),
            }
        ],
    }
    response = (call or _call_claude)(body)

    tool_input: dict | None = None
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "emit_meeting_notes":
            tool_input = block.get("input")
            break
    if tool_input is None:
        raise RuntimeError("LLM did not return the expected tool_use block")

    result = _enforce_rules(tool_input, utterances, roster)
    log.info(
        "stage=extract elapsed_s=%.3f tasks=%d",
        time.perf_counter() - t0,
        len(result["tasks"]),
    )
    return result
