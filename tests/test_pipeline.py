from pathlib import Path

from app.feedback import FeedbackStore
from app.pipeline import MeetingResult, run_pipeline


def _scribe_response(text: str) -> dict:
    words: list[dict] = []
    t = 0.0
    tokens = text.split()
    for i, w in enumerate(tokens):
        words.append(
            {"type": "word", "text": w, "start": t, "end": t + 0.3, "speaker_id": "speaker_0"}
        )
        if i < len(tokens) - 1:
            words.append(
                {
                    "type": "spacing", "text": " ",
                    "start": t + 0.3, "end": t + 0.4, "speaker_id": "speaker_0",
                }
            )
        t += 0.4
    return {"language_code": "hin", "words": words}


def test_run_pipeline_returns_meeting_result(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    store = FeedbackStore(tmp_path / "fb.db")

    def fake_post(audio_path: Path, keyterms) -> dict:
        return _scribe_response("Kaushal please prep the ppt")

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {
            "summary": "ok",
            "notes": [{"topic": "Planets", "points": ["Sun ek dwarf star hai"]}],
            "tasks": [
                {
                    "assignee": "Kaushal", "assigned_by": "speaker_0",
                    "task": "ppt", "deadline": None,
                    "source_quote": "Kaushal please prep the ppt", "confidence": 0.9,
                }
            ],
        }

    result = run_pipeline(
        audio, "demo", ["Kaushal"], store,
        post=fake_post, call=fake_call, use_cache=False,
    )
    assert isinstance(result, MeetingResult)
    assert result.audio_path == audio
    assert result.team_id == "demo"
    assert result.roster == ["Kaushal"]
    assert result.summary == "ok"
    assert result.tasks and result.tasks[0]["assignee"] == "Kaushal"
    assert set(result.stage_seconds) == {"stt", "speakers", "extract"}
    assert result.total_seconds > 0
    # Timing logs from the three modules + pipeline_inputs's feedback_lookup.
    assert any("stage=stt" in line for line in result.stage_logs)
    assert any("stage=feedback_lookup" in line for line in result.stage_logs)
    assert any("stage=extract" in line for line in result.stage_logs)


def test_run_pipeline_feeds_glossary_as_keyterms_to_stt(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("demo", "Kaushal", "person")

    received: list[list[str]] = []

    def fake_post(audio_path: Path, keyterms) -> dict:
        received.append(list(keyterms))
        return {"language_code": "hin", "words": []}

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {"summary": "", "notes": [], "tasks": []}

    run_pipeline(audio, "demo", [], store, post=fake_post, call=fake_call, use_cache=False)
    assert received == [["Kaushal"]]


def test_run_pipeline_pulls_top_examples_into_extract(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")
    store = FeedbackStore(tmp_path / "fb.db")
    store.correct_task(
        "demo",
        {"assignee": "Koshal", "task": "ppt"},
        {"assignee": "Kaushal", "task": "ppt"},
        "Koshal please prep the ppt",
    )

    seen: list[list] = []

    def fake_post(audio_path: Path, keyterms) -> dict:
        return _scribe_response("Koshal please prep the ppt")

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {"summary": "", "notes": [], "tasks": []}

    def fake_extract(utterances, roster, **kwargs):
        seen.append(kwargs.get("examples", []))
        return {"summary": "", "notes": [], "tasks": []}

    import pytest

    monkey = pytest.MonkeyPatch()
    monkey.setattr("app.pipeline.extract", fake_extract)
    try:
        run_pipeline(
            audio, "demo", ["Kaushal"], store,
            post=fake_post, call=fake_call, use_cache=False,
        )
    finally:
        monkey.undo()

    assert seen and seen[0]
    assert seen[0][0]["after"] == {"assignee": "Kaushal", "task": "ppt"}


def test_run_pipeline_tolerates_none_store(tmp_path):
    """Bench's test path passes store=None; pipeline must still run and return a result."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")

    def fake_post(audio_path: Path, keyterms) -> dict:
        assert list(keyterms) == []  # no glossary available without a store
        return {"language_code": "hin", "words": []}

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {"summary": "", "notes": [], "tasks": []}

    result = run_pipeline(
        audio, "demo", ["A"], None,
        post=fake_post, call=fake_call, use_cache=False,
    )
    assert isinstance(result, MeetingResult)
    assert result.total_seconds > 0


def test_meeting_result_to_dict_round_trips_utterances(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")
    store = FeedbackStore(tmp_path / "fb.db")

    def fake_post(audio_path: Path, keyterms) -> dict:
        return _scribe_response("hi there")

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {"summary": "s", "notes": [], "tasks": []}

    result = run_pipeline(
        audio, "demo", [], store,
        post=fake_post, call=fake_call, use_cache=False,
    )
    d = result.to_dict()
    assert d["summary"] == "s"
    assert d["audio_path"] == str(audio)
    assert isinstance(d["utterances"], list)
    assert d["utterances"][0]["text"] == "hi there"
