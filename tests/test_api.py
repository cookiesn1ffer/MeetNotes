from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.feedback import FeedbackStore
from app.models import Utterance


@pytest.fixture
def client(tmp_path) -> TestClient:
    store = FeedbackStore(tmp_path / "fb.db")
    return TestClient(create_app(store))


def test_correct_task_endpoint_persists(client):
    r = client.post(
        "/correct_task",
        json={
            "team_id": "default",
            "before": {"task": "wrong", "assignee": "Someone"},
            "after": {"task": "ppt on planets", "assignee": "Kaushal"},
            "source_quote": "please prep the ppt on planets",
        },
    )
    assert r.status_code == 200
    assert isinstance(r.json()["id"], int)
    store = client.app.state.store
    examples = store.retrieve_examples("default", "ppt on planets")
    assert examples[0]["after"]["task"] == "ppt on planets"


def test_correct_speaker_endpoint_persists(client, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake")
    r = client.post(
        "/correct_speaker",
        json={"before": "speaker_0", "after": "Aarush", "clip_path": str(clip)},
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.enrollments_for("default", "Aarush") == [Path(clip)]


def test_correct_speaker_endpoint_without_clip(client):
    r = client.post(
        "/correct_speaker",
        json={"before": "speaker_0", "after": "Aarush"},
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.enrollments_for("default", "Aarush") == []


def test_correct_speaker_endpoint_with_embedding(client):
    r = client.post(
        "/correct_speaker",
        json={"before": "speaker_0", "after": "Aarush", "embedding": [0.5, 0.5]},
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.get_speaker_embedding("default", "Aarush") == [0.5, 0.5]


def test_correct_speaker_endpoint_with_audio_span(client, tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def fake_ecapa(cache_dir=None):
        return (lambda path, start, end: [0.1, 0.9]), (lambda path: [0.1, 0.9])

    monkeypatch.setattr("app.feedback.ecapa_embedder", fake_ecapa)
    monkeypatch.setattr("app.feedback._default_ecapa_embedder", None)
    r = client.post(
        "/correct_speaker",
        json={
            "before": "speaker_0", "after": "Aarush",
            "audio_path": str(audio), "start": 0.0, "end": 2.0,
        },
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.get_speaker_embedding("default", "Aarush") == [0.1, 0.9]


def test_add_glossary_term_endpoint(client):
    r = client.post("/glossary", json={"term": "OKR", "kind": "acronym"})
    assert r.status_code == 200
    store = client.app.state.store
    assert store.glossary("default") == {"OKR": "acronym"}


def test_add_glossary_term_endpoint_team_scoping(client):
    client.post("/glossary", json={"team_id": "teamA", "term": "Kaushal", "kind": "person"})
    client.post("/glossary", json={"team_id": "teamB", "term": "OKR", "kind": "acronym"})
    store = client.app.state.store
    assert store.keyterms("teamA") == ["Kaushal"]
    assert store.keyterms("teamB") == ["OKR"]


# ---------------------------------------------------------------- UI + pipeline + TTS


def test_index_route_serves_fallback_html_without_built_ui(tmp_path):
    client = TestClient(create_app(FeedbackStore(tmp_path / "fb.db"), ui_dir=tmp_path / "no-ui"))
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<title>MeetNotes</title>" in body
    assert '/process' in body
    assert '/correct_task' in body
    assert '/tts' in body


def test_process_endpoint_runs_pipeline_and_returns_timings(client, monkeypatch):
    def fake_transcribe(audio, **kwargs):
        # Prove the glossary wiring reached stt via keyterms (not team_id anymore).
        assert isinstance(kwargs.get("keyterms"), list)
        return [Utterance("speaker_0", 0.0, 2.0, "Kaushal, make a ppt", "hin")]

    def fake_extract(utterances, roster, **kwargs):
        assert isinstance(kwargs.get("glossary"), dict)
        assert isinstance(kwargs.get("examples"), list)
        return {
            "summary": "ok",
            "notes": [],
            "tasks": [
                {
                    "assignee": "Kaushal",
                    "assigned_by": "speaker_0",
                    "task": "ppt on planets",
                    "deadline": None,
                    "source_quote": "Kaushal, make a ppt",
                    "confidence": 0.9,
                }
            ],
        }

    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.extract", fake_extract)

    r = client.post(
        "/process",
        data={"team_id": "demo", "roster": "Aarush, Kaushal"},
        files={"audio": ("clip.wav", b"audio-bytes", "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "ok"
    assert data["tasks"][0]["assignee"] == "Kaushal"
    assert data["roster"] == ["Aarush", "Kaushal"]
    assert isinstance(data["timings"], list)
    assert data["utterances"][0]["text"] == "Kaushal, make a ppt"


def test_process_rejects_empty_upload(client):
    r = client.post(
        "/process",
        data={"team_id": "demo"},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400


def test_tts_endpoint_proxies_elevenlabs(client, monkeypatch):
    received: dict = {}

    def fake_fetch(voice_id: str, text: str) -> bytes:
        received["voice_id"] = voice_id
        received["text"] = text
        return b"\xff\xfb\x90mp3-bytes"

    monkeypatch.setattr("app.api.main._tts_fetch", fake_fetch)
    r = client.post("/tts", json={"text": "hello, world"})
    assert r.status_code == 200
    assert r.content.startswith(b"\xff\xfb")
    assert r.headers["content-type"] == "audio/mpeg"
    assert received["text"] == "hello, world"
    # Default voice id when neither request nor env sets one.
    assert received["voice_id"]


def test_tts_endpoint_rejects_empty_text(client, monkeypatch):
    monkeypatch.setattr("app.api.main._tts_fetch", lambda *a, **k: b"")
    r = client.post("/tts", json={"text": "   "})
    assert r.status_code == 400


def test_tts_endpoint_voice_id_override(client, monkeypatch):
    received: dict = {}

    def fake_fetch(voice_id, text):
        received["voice_id"] = voice_id
        return b"ok"

    monkeypatch.setattr("app.api.main._tts_fetch", fake_fetch)
    r = client.post("/tts", json={"text": "x", "voice_id": "voice-123"})
    assert r.status_code == 200
    assert received["voice_id"] == "voice-123"


def test_correction_persists_across_process_calls(client, monkeypatch):
    """End-to-end PASS: process -> correct_task -> process again, the correction
    reaches the second extract call as a top-3 few-shot example (via run_pipeline
    -> pipeline_inputs -> retrieve_examples)."""
    seen_examples: list[list] = []

    def fake_transcribe(audio, **kwargs):
        return [Utterance("speaker_0", 0.0, 2.0, "Koshal please prep the ppt", "hin")]

    def fake_extract(utterances, roster, **kwargs):
        seen_examples.append(kwargs.get("examples", []))
        return {"summary": "", "notes": [], "tasks": []}

    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.extract", fake_extract)

    files = {"audio": ("c.wav", b"a", "audio/wav")}
    r1 = client.post("/process", data={"team_id": "demo", "roster": "Kaushal"}, files=files)
    assert r1.status_code == 200
    assert seen_examples[0] == []  # nothing stored yet

    cr = client.post(
        "/correct_task",
        json={
            "team_id": "demo",
            "before": {"assignee": "Koshal", "task": "ppt"},
            "after": {"assignee": "Kaushal", "task": "ppt"},
            "source_quote": "Koshal please prep the ppt",
        },
    )
    assert cr.status_code == 200

    r2 = client.post("/process", data={"team_id": "demo", "roster": "Kaushal"}, files=files)
    assert r2.status_code == 200
    # Second extract run SAW the correction via retrieve_examples.
    assert seen_examples[1]
    assert seen_examples[1][0]["after"] == {"assignee": "Kaushal", "task": "ppt"}


def test_lazy_module_app_is_created_on_attribute_access(monkeypatch, tmp_path):
    """Importing app.api.main must not eagerly create a FeedbackStore at DEFAULT_DB."""
    import app.api.main as mod

    # Make DEFAULT_DB point into tmp so if accidentally created, it lands there.
    monkeypatch.setattr(mod, "DEFAULT_DB", tmp_path / "lazy.db")
    monkeypatch.setattr(mod, "_lazy_app", None)
    assert not (tmp_path / "lazy.db").exists()
    _ = mod.app  # triggers __getattr__
    assert (tmp_path / "lazy.db").exists()
