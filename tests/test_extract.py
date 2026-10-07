import pytest

from app.extract import _dispatch, _match_assignee, extract
from app.feedback import FeedbackStore
from app.models import Utterance

SENTENCE = (
    "Sun ek dwarf star hai. "
    "Kaushal you are assigned to make a ppt on planets for next meeting"
)


def _utts():
    return [Utterance("Aarush", 0.0, 5.0, SENTENCE, "hin")]


def _payload(**overrides):
    base = {"summary": "s", "notes": [], "tasks": []}
    base.update(overrides)
    return base


def _task(**overrides):
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


# ---------------------------------------------------------------- rule enforcement


def test_example_sentence_extracts_single_task():
    captured: dict = {}

    def fake(system, user, schema):
        captured["system"] = system
        captured["user"] = user
        captured["schema"] = schema
        return _payload(tasks=[_task()])

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert len(result["tasks"]) == 1
    task = result["tasks"][0]
    assert task["assignee"] == "Kaushal"
    assert task["assigned_by"] == "Aarush"
    assert task["task"] == "ppt on planets"
    assert task["deadline"] == "next meeting"
    assert task["source_quote"] in SENTENCE
    # Schema was handed to the provider.
    assert captured["schema"]["properties"]["tasks"]["items"]["required"]


def test_fuzzy_assignee_koshal_matches_kaushal():
    def fake(system, user, schema):
        return _payload(tasks=[_task(assignee="Koshal")])

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"][0]["assignee"] == "Kaushal"


def test_task_without_source_quote_in_transcript_is_dropped():
    def fake(system, user, schema):
        return _payload(tasks=[_task(source_quote="not in the transcript")])

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"] == []


def test_assignee_not_in_roster_is_dropped():
    def fake(system, user, schema):
        return _payload(
            tasks=[_task(assignee="Totally Different", source_quote="Sun ek dwarf star hai.")]
        )

    result = extract(_utts(), roster=["Aarush", "Kaushal"], call=fake)
    assert result["tasks"] == []


def test_exactly_one_llm_call():
    calls = []

    def fake(system, user, schema):
        calls.append(1)
        return _payload()

    extract(_utts(), roster=["Aarush"], call=fake)
    assert len(calls) == 1


def test_match_assignee_handles_devanagari_transliteration():
    assert _match_assignee("कौशल", ["Aarush", "Kaushal"]) == "Kaushal"


def test_match_assignee_rejects_far_names():
    assert _match_assignee("Zephyr", ["Aarush", "Kaushal"]) is None


# ---------------------------------------------------------------- prompt wiring


def test_prompt_includes_roster_glossary_and_examples_passed_inline():
    def fake(system, user, schema):
        assert "Kaushal" in user
        assert "KPI — acronym" in user
        assert "ppt on planets" in user
        return _payload()

    extract(
        _utts(),
        roster=["Aarush", "Kaushal"],
        glossary={"KPI": "acronym"},
        examples=[{"before": {}, "after": {"task": "ppt on planets"}, "source_quote": "x"}],
        call=fake,
    )


def test_team_glossary_from_store_appears_in_prompt(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("teamA", "Kaushal", "person")
    seen: dict = {}

    def fake(system, user, schema):
        seen["user"] = user
        return _payload()

    extract(_utts(), roster=["Aarush", "Kaushal"], team_id="teamA", store=store, call=fake)
    assert "Kaushal — person" in seen["user"]


def test_team_glossary_scoped_by_team_id_in_extract(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("teamA", "Kaushal", "person")
    seen: dict = {}

    def fake(system, user, schema):
        seen["user"] = user
        return _payload()

    extract(_utts(), roster=["Aarush", "Kaushal"], team_id="teamB", store=store, call=fake)
    assert "Kaushal — person" not in seen["user"]


def test_examples_auto_fetched_from_store(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.correct_task(
        "teamA",
        {"task": "wrong"},
        {"task": "ppt on planets"},
        "Kaushal you are assigned to make a ppt on planets",
    )
    seen: dict = {}

    def fake(system, user, schema):
        seen["user"] = user
        return _payload()

    extract(_utts(), roster=["Aarush", "Kaushal"], team_id="teamA", store=store, call=fake)
    assert "ppt on planets" in seen["user"]


def test_koshal_correction_changes_extract_output_on_similar_transcript(tmp_path):
    """Stored 'Koshal' -> 'Kaushal' task correction surfaces as a few-shot example
    whose wrong AND corrected forms appear verbatim in an echoing LLM's output.

    The LLM mock echoes the few-shot block it received into the summary, so if the
    retrieval wired a past correction into the prompt, the correction's two forms
    land in the summary; otherwise the summary mentions neither.
    """
    quote = "Koshal you are assigned to make a ppt on planets for next meeting"
    utts = [Utterance("Aarush", 0.0, 5.0, quote, "hin")]

    def echo_call(system: str, user: str, schema: dict) -> dict:
        # Return the correction-examples block verbatim as the summary.
        header = "Correction examples (wrong output -> corrected output):"
        tail = "\n\nTranscript:"
        start = user.index(header)
        end = user.index(tail, start)
        return {"summary": user[start:end], "notes": [], "tasks": []}

    # Empty store -> no example in the prompt, so neither spelling leaks into summary.
    empty = FeedbackStore(tmp_path / "empty.db")
    out_empty = extract(utts, roster=["Aarush", "Kaushal"], team_id="t", store=empty, call=echo_call)
    assert "Koshal" not in out_empty["summary"]
    assert "Kaushal" not in out_empty["summary"]

    # After storing the correction, both spellings appear via the retrieved example.
    store = FeedbackStore(tmp_path / "fb.db")
    store.correct_task(
        "t",
        {"assignee": "Koshal", "task": "ppt on planets"},
        {"assignee": "Kaushal", "task": "ppt on planets"},
        quote,
    )
    out_with = extract(
        utts, roster=["Aarush", "Kaushal"], team_id="t", store=store, call=echo_call
    )
    assert "Koshal" in out_with["summary"]
    assert "Kaushal" in out_with["summary"]
    assert out_with["summary"] != out_empty["summary"]


def test_retrieve_examples_returns_empty_without_corrections(tmp_path):
    store = FeedbackStore(tmp_path / "empty.db")
    assert store.retrieve_examples("t", "anything") == []


# ---------------------------------------------------------------- LLM provider dispatch


def test_llm_provider_gemini_default_routes_to_gemini(monkeypatch):
    calls: list[str] = []
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    def fake_gemini(system, user, schema):
        calls.append("gemini")
        return _payload()

    def fake_claude(system, user, schema):
        calls.append("anthropic")
        return _payload()

    monkeypatch.setattr("app.extract._call_gemini", fake_gemini)
    monkeypatch.setattr("app.extract._call_claude", fake_claude)
    extract(_utts(), roster=["Aarush"])
    assert calls == ["gemini"]


def test_llm_provider_anthropic_env_routes_to_claude(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    def fake_gemini(system, user, schema):
        calls.append("gemini")
        return _payload()

    def fake_claude(system, user, schema):
        calls.append("anthropic")
        return _payload()

    monkeypatch.setattr("app.extract._call_gemini", fake_gemini)
    monkeypatch.setattr("app.extract._call_claude", fake_claude)
    extract(_utts(), roster=["Aarush"])
    assert calls == ["anthropic"]


def test_provider_kwarg_overrides_env(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    def fake_gemini(system, user, schema):
        calls.append("gemini")
        return _payload()

    monkeypatch.setattr("app.extract._call_gemini", fake_gemini)
    extract(_utts(), roster=["Aarush"], provider="gemini")
    assert calls == ["gemini"]


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        _dispatch("sys", "user", {})


# ---------------------------------------------------------------- openai-compatible provider


def _capture_post(monkeypatch, content):
    seen = {}

    def fake_post(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("app.extract._post", fake_post)
    return seen


def test_openai_compat_defaults_to_opencode_go_deepseek(monkeypatch):
    from app.extract import SCHEMA, _call_openai_compat

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "")  # as copied from .env.example: empty means default
    monkeypatch.setenv("LLM_MODEL", "")
    seen = _capture_post(monkeypatch, '{"summary": "s", "notes": [], "tasks": []}')
    assert _call_openai_compat("sys", "user", SCHEMA)["summary"] == "s"
    assert seen["url"] == "https://opencode.ai/zen/go/v1/chat/completions"
    assert seen["body"]["model"] == "deepseek-v4.1-flash"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["headers"]["authorization"] == "Bearer k"
    assert "JSON schema" in seen["body"]["messages"][0]["content"]


def test_openai_compat_env_overrides_and_fenced_json(monkeypatch):
    from app.extract import _call_openai_compat

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("LLM_MODEL", "m-1")
    seen = _capture_post(monkeypatch, '```json\n{"summary": "x", "notes": [], "tasks": []}\n```')
    assert _call_openai_compat("s", "u", {})["summary"] == "x"
    assert seen["url"].startswith("https://example.test") and seen["body"]["model"] == "m-1"


def test_openai_compat_bad_responses_raise_clean_errors(monkeypatch):
    from app.extract import _call_openai_compat

    monkeypatch.setenv("LLM_API_KEY", "k")
    _capture_post(monkeypatch, "not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _call_openai_compat("s", "u", {})
    monkeypatch.setattr("app.extract._post", lambda *a: {"error": "x"})
    with pytest.raises(RuntimeError, match="no message content"):
        _call_openai_compat("s", "u", {})


def test_openai_compat_requires_key(monkeypatch, tmp_path):
    from app.extract import _call_openai_compat

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        _call_openai_compat("s", "u", {})


@pytest.mark.parametrize("name", ["openai_compat", "opencode"])
def test_dispatch_routes_openai_compat(monkeypatch, name):
    monkeypatch.setenv("LLM_PROVIDER", name)
    monkeypatch.setattr("app.extract._call_openai_compat", lambda s, u, sc: {"summary": "ok"})
    assert _dispatch("s", "u", {}) == {"summary": "ok"}
