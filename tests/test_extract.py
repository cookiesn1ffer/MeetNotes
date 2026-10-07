import pytest

from app.extract import _match_assignee, extract
from app.models import Utterance

SENTENCE = (
    "Sun ek dwarf star hai. "
    "Kaushal you are assigned to make a ppt on planets for next meeting"
)


def _utts():
    return [Utterance("Aarush", 0.0, 5.0, SENTENCE, "hin")]


def _tool_use(payload: dict) -> dict:
    return {"content": [{"type": "tool_use", "name": "emit_meeting_notes", "input": payload}]}


def _task(**overrides) -> dict:
    base = {
        "assignee": "Kaushal",
        "assigned_by": "Aarush",
        "task": "ppt on planets",
        "deadline": "next meeting",
        "source_quote": (
            "Kaushal you are assigned to make a ppt on planets for next meeting"
        ),
        "confidence": 0.95,
    }
    base.update(overrides)
    return base


def test_example_sentence_extracts_single_task():
    captured: dict = {}

    def fake(body):
        captured["body"] = body
        return _tool_use({"summary": "s", "notes": [], "tasks": [_task()]})

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert len(result["tasks"]) == 1
    task = result["tasks"][0]
    assert task["assignee"] == "Kaushal"
    assert task["assigned_by"] == "Aarush"
    assert task["task"] == "ppt on planets"
    assert task["deadline"] == "next meeting"
    assert task["source_quote"] in SENTENCE
    # Call shape: one tool-forced invocation.
    assert captured["body"]["tool_choice"] == {"type": "tool", "name": "emit_meeting_notes"}


def test_fuzzy_assignee_koshal_matches_kaushal():
    def fake(body):
        return _tool_use({"summary": "", "notes": [], "tasks": [_task(assignee="Koshal")]})

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"][0]["assignee"] == "Kaushal"


def test_task_without_source_quote_in_transcript_is_dropped():
    def fake(body):
        return _tool_use(
            {
                "summary": "",
                "notes": [],
                "tasks": [_task(source_quote="not in the transcript")],
            }
        )

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"] == []


def test_assignee_not_in_roster_is_dropped():
    def fake(body):
        return _tool_use(
            {
                "summary": "",
                "notes": [],
                "tasks": [_task(assignee="Totally Different", source_quote="Sun ek dwarf star hai.")],
            }
        )

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"] == []


def test_exactly_one_llm_call():
    calls = []

    def fake(body):
        calls.append(body)
        return _tool_use({"summary": "", "notes": [], "tasks": []})

    extract(_utts(), roster=["Aarush"], call=fake)
    assert len(calls) == 1


def test_prompt_includes_roster_glossary_and_examples():
    def fake(body):
        content = body["messages"][0]["content"]
        assert "Kaushal" in content
        assert "ppt: PowerPoint presentation" in content
        assert "prior-task-id" in content
        return _tool_use({"summary": "", "notes": [], "tasks": []})

    extract(
        _utts(),
        roster=["Aarush", "Kaushal"],
        glossary={"ppt": "PowerPoint presentation"},
        examples=[{"id": "prior-task-id", "task": "..."}],
        call=fake,
    )


def test_only_first_three_examples_are_sent():
    def fake(body):
        content = body["messages"][0]["content"]
        assert "ex4" not in content
        assert "ex3" in content
        return _tool_use({"summary": "", "notes": [], "tasks": []})

    extract(
        _utts(),
        roster=["Aarush"],
        examples=[{"id": f"ex{i}"} for i in range(1, 5)],
        call=fake,
    )


def test_missing_tool_use_block_raises():
    def fake(body):
        return {"content": [{"type": "text", "text": "nope"}]}

    with pytest.raises(RuntimeError, match="tool_use"):
        extract(_utts(), roster=["Aarush"], call=fake)


def test_match_assignee_handles_devanagari_transliteration():
    assert _match_assignee("कौशल", ["Aarush", "Kaushal"]) == "Kaushal"


def test_match_assignee_rejects_far_names():
    assert _match_assignee("Zephyr", ["Aarush", "Kaushal"]) is None
