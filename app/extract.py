"""Single-LLM-call extraction of summary, notes, and tasks from a resolved transcript.

One call to the LLM selected by LLM_PROVIDER (`gemini` or `anthropic`, default `gemini`).
Tool-use / response_schema forces JSON that matches SCHEMA. The HTTP call for each
provider is a module-level function and can be monkeypatched in tests; the public
`extract()` also accepts an injected `call` for full offline control.

Post-processing enforces the prompt's rules as defense in depth:
  * tasks without a `source_quote` that appears in the transcript are dropped
  * `assignee` is fuzzy-matched to the roster (handles Kaushal/Koshal, कौशल, etc.)
"""

import difflib
import json
import logging
import os
import time
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.config import get_anthropic_key, get_gemini_key
from app.models import Utterance
from app.stt import to_roman

if TYPE_CHECKING:
    from app.feedback import FeedbackStore

log = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5-5",
}

NAME_MATCH_CUTOFF = 0.7  # difflib ratio; "koshal" vs "kaushal" is ~0.77

LLMCall = Callable[[str, str, dict], dict]  # (system, user, schema) -> parsed payload


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
- Emit exactly one structured response matching the schema.
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
    """Build the user-content prompt. `glossary` maps term -> kind (person, acronym, ...)."""
    lines: list[str] = ["Roster (assignees must be from this list):"]
    lines.append("  " + (", ".join(roster) if roster else "(none)"))
    lines += ["", "Team glossary (term — kind; preserve the exact spelling):"]
    if glossary:
        lines += [f"  {term} — {kind}" for term, kind in glossary.items()]
    else:
        lines.append("  (none)")
    lines += ["", "Correction examples (past tasks and their accepted extraction):"]
    if examples:
        lines += ["  " + json.dumps(ex, ensure_ascii=False) for ex in examples[:3]]
    else:
        lines.append("  (none)")
    lines += ["", "Transcript:", _render_transcript(utterances)]
    return "\n".join(lines)


def _post(url: str, headers: dict[str, str], body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_claude(system: str, user: str, schema: dict) -> dict:
    """Anthropic Messages API; tool-use forces structured output. Returns the parsed input."""
    body = {
        "model": DEFAULT_MODELS["anthropic"],
        "max_tokens": 4096,
        "system": system,
        "tools": [
            {
                "name": "emit_meeting_notes",
                "description": "Emit the structured meeting output.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_meeting_notes"},
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": get_anthropic_key(),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    response = _post(ANTHROPIC_MESSAGES_URL, headers, body)
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "emit_meeting_notes":
            return block.get("input") or {}
    msg = "Anthropic response did not contain the expected tool_use block"
    raise RuntimeError(msg)


def _call_gemini(system: str, user: str, schema: dict) -> dict:
    """Google Generative Language API; response_schema forces structured JSON output."""
    key = get_gemini_key()
    model = DEFAULT_MODELS["gemini"]
    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={key}"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generation_config": {
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    }
    response = _post(url, {"content-type": "application/json"}, body)
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        msg = "Gemini response had no candidate text"
        raise RuntimeError(msg) from exc
    return json.loads(text)


def _dispatch(system: str, user: str, schema: dict, provider: str | None = None) -> dict:
    """Pick a provider via arg or LLM_PROVIDER env (default 'gemini'). Looks up by name,
    so monkeypatching `_call_claude`/`_call_gemini` on the module works."""
    p = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if p == "gemini":
        return _call_gemini(system, user, schema)
    if p == "anthropic":
        return _call_claude(system, user, schema)
    msg = f"Unknown LLM_PROVIDER={p!r}; expected 'gemini' or 'anthropic'"
    raise RuntimeError(msg)


def _match_assignee(name: str, roster: list[str]) -> str | None:
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
    team_id: str | None = None,
    store: "FeedbackStore | None" = None,
    provider: str | None = None,
    call: LLMCall | None = None,
) -> dict:
    """One LLM call. Returns {"summary", "notes", "tasks"} after rule enforcement.

    If `team_id` + `store` are given, glossary terms for that team are merged in and,
    if `examples` is None, the top-3 TF-IDF-similar past corrections are pulled in.
    `provider` overrides the LLM_PROVIDER env var. `call` fully overrides dispatch.
    """
    t0 = time.perf_counter()
    merged_glossary = dict(glossary or {})
    merged_examples = list(examples) if examples is not None else []
    if team_id and store is not None:
        for term, kind in store.glossary(team_id).items():
            merged_glossary.setdefault(term, kind)
        if examples is None:
            transcript = " ".join(u.text for u in utterances)
            merged_examples = store.top_examples(team_id, transcript, k=3)

    system = SYSTEM_RULES
    user = _prompt(utterances, roster, merged_glossary, merged_examples)
    if call is not None:
        payload = call(system, user, SCHEMA)
    else:
        payload = _dispatch(system, user, SCHEMA, provider=provider)

    result = _enforce_rules(payload, utterances, roster)
    log.info(
        "stage=extract elapsed_s=%.3f tasks=%d",
        time.perf_counter() - t0,
        len(result["tasks"]),
    )
    return result
