"""Contract tests: every endpoint is hit with the exact payload shape ui/src/api.js sends,
and /process is checked for every key the UI reads."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from app import cli
from app.api.main import create_app
from app.api.shape import line_language, mmss
from app.feedback import FeedbackStore
from app.models import Utterance

TASK = {
    "assignee": "Kaushal",
    "assigned_by": "Aarush",
    "task": "ppt on planets",
    "deadline": "next meeting",
    "source_quote": "Kaushal make a ppt on planets",
    "confidence": 0.9,
}


@pytest.fixture
def ui_dir(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="app"></div>', encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return dist


@pytest.fixture
def client(tmp_path, ui_dir):
    return TestClient(create_app(FeedbackStore(tmp_path / "data" / "fb.db"), ui_dir=ui_dir))


@pytest.fixture
def fake_pipeline(monkeypatch):
    def fake_transcribe(audio, **kwargs):
        return [
            Utterance("speaker_0", 0.0, 4.0, "Hi, I'm Aarush", "eng"),
            Utterance("speaker_1", 65.0, 69.0, "कल meeting है", "hin"),
        ]

    def fake_extract(utterances, roster, **kwargs):
        return {
            "summary": "Planning call.",
            "notes": [
                {"topic": "Planets", "points": ["Sun is a dwarf star"], "person": "Kaushal"},
                {"topic": "General", "points": ["Next meeting Friday"]},
            ],
            "tasks": [TASK],
        }

    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("app.pipeline.extract", fake_extract)


def _process(client, **overrides):
    # Exactly what api.js processMeeting() sends.
    data = {"team_id": "Product / India", "roster": "Aarush,Kaushal"}
    data.update(overrides)
    return client.post(
        "/process", data=data, files={"audio": ("clip.webm", b"audio-bytes", "audio/webm")}
    )


# ---------------------------------------------------------------- UI hosting + CORS


def test_built_ui_is_served_at_root_and_api_routes_still_work(client):
    root = client.get("/")
    assert root.status_code == 200 and '<div id="app">' in root.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.post("/glossary", json={"term": "OKR", "kind": "term"}).status_code == 200


def test_cors_allows_localhost_dev_server_only(client):
    ok = client.options(
        "/process",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    bad = client.options(
        "/process",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


# ---------------------------------------------------------------- /process


def test_process_returns_everything_the_ui_reads(client, fake_pipeline):
    r = _process(client)
    assert r.status_code == 200
    data = r.json()

    assert re.fullmatch(r"[0-9a-f]{12}", data["run_id"])
    assert data["confidence"] == 50.0  # Aarush named for 4s, speaker_1 unknown for 4s
    assert set(data["stage_timings"]) == {"transcribe", "speakers", "tasks", "total"}

    names = [p["name"] for p in data["people"]]
    assert names == ["Aarush", "Kaushal", "Meeting"]
    kaushal = data["people"][1]
    assert kaushal["id"] and "Planets: Sun is a dwarf star" in kaushal["notes"]
    task = kaushal["tasks"][0]
    assert {"id", "task", "assigned_by", "assignee", "deadline", "source_quote"} <= set(task)
    assert data["people"][0]["tasks"] == []
    meeting = data["people"][2]
    assert "Planning call." in meeting["notes"] and "Next meeting Friday" in meeting["notes"]

    first, second = data["transcript"]
    assert {"id", "speaker", "start", "end", "timestamp", "text", "language", "lang"} <= set(first)
    assert (first["speaker"], first["language"]) == ("Aarush", "EN")
    assert second["speaker"] == "Unknown 1"
    assert (second["language"], second["timestamp"]) == ("HI + EN", "01:05")

    # Fields that existed before the UI integration are kept.
    assert data["tasks"][0]["id"] == task["id"] and data["roster"] == ["Aarush", "Kaushal"]
    assert isinstance(data["timings"], list) and data["utterances"][0]["lang"] == "eng"


def test_process_keeps_upload_for_later_voice_learning(client, fake_pipeline, tmp_path):
    run_id = _process(client).json()["run_id"]
    assert list((tmp_path / "data" / "uploads").glob(f"{run_id}.*"))


def test_process_errors_are_one_line_json(client, monkeypatch):
    empty = client.post("/process", data={"team_id": "t"}, files={"audio": ("e.wav", b"", "audio/wav")})
    assert empty.status_code == 400 and empty.json() == {"detail": "empty audio upload"}

    def no_key(audio, **kwargs):
        raise RuntimeError("ELEVENLABS_API_KEY is not set (put it in .env)")

    monkeypatch.setattr("app.pipeline.transcribe", no_key)
    missing = _process(client)
    assert missing.status_code == 503 and "ELEVENLABS_API_KEY" in missing.json()["detail"]

    import urllib.error

    def down(audio, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("app.pipeline.transcribe", down)
    net = _process(client)
    assert net.status_code == 502 and "connection refused" in net.json()["detail"]

    def bad_file(audio, **kwargs):
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)

    monkeypatch.setattr("app.pipeline.transcribe", bad_file)
    bad = _process(client)
    assert bad.status_code == 502 and "400" in bad.json()["detail"]


# ---------------------------------------------------------------- corrections / glossary / tts


def test_correct_task_with_ui_payload(client):
    # api.js correctTask(): toBackendTask() on both sides, deadline "—" -> null.
    payload = {
        "team_id": "Product / India",
        "before": {"task": "ppt", "assigned_by": "Aarush", "assignee": "Koshal", "deadline": None},
        "after": {"task": "ppt", "assigned_by": "Aarush", "assignee": "Kaushal", "deadline": None},
        "source_quote": "Koshal make a ppt",
    }
    assert client.post("/correct_task", json=payload).status_code == 200
    examples = client.app.state.store.retrieve_examples("Product / India", "Koshal make a ppt")
    assert examples[0]["after"]["assignee"] == "Kaushal"


def test_correct_speaker_learns_voice_from_uploaded_run(client, fake_pipeline, monkeypatch):
    run_id = _process(client).json()["run_id"]
    spans = []

    def fake_embed(audio, start, end, embedder=None):
        spans.append((audio.name.split(".")[0], start, end))
        return [0.0, 1.0]

    monkeypatch.setattr(FeedbackStore, "embed_span", lambda self, a, s, e, embedder=None: fake_embed(a, s, e))
    payload = {  # api.js correctSpeaker()
        "team_id": "Product / India",
        "before": "Unknown 1",
        "after": "Kaushal",
        "run_id": run_id,
        "start": 65.0,
        "end": 69.0,
    }
    r = client.post("/correct_speaker", json=payload)
    assert r.status_code == 200 and r.json()["voice_learned"] is True
    assert spans == [(run_id, 65.0, 69.0)]
    assert client.app.state.store.get_speaker_embedding("Product / India", "Kaushal") == [0.0, 1.0]


def test_correct_speaker_without_embeddings_still_records_the_name(client, fake_pipeline, monkeypatch):
    run_id = _process(client).json()["run_id"]
    monkeypatch.setattr(FeedbackStore, "embed_span", lambda self, a, s, e, embedder=None: None)
    r = client.post(
        "/correct_speaker",
        json={"team_id": "t", "before": "Unknown 1", "after": "Kaushal", "run_id": run_id, "start": 0, "end": 1},
    )
    assert r.status_code == 200 and r.json()["voice_learned"] is False
    assert "Kaushal" in client.app.state.store.build_roster("t", {}).enrollments


def test_correct_speaker_rejects_path_tricks_in_run_id(client, monkeypatch):
    seen = []
    monkeypatch.setattr(FeedbackStore, "embed_span", lambda self, a, s, e, embedder=None: seen.append(a))
    r = client.post(
        "/correct_speaker",
        json={"team_id": "t", "before": "x", "after": "Y", "run_id": "../../etc/passwd", "start": 0, "end": 1},
    )
    assert r.status_code == 200 and seen == []


def test_glossary_with_ui_payload(client):
    r = client.post("/glossary", json={"team_id": "Product / India", "term": "Kaushal", "kind": "term"})
    assert r.status_code == 200
    assert client.app.state.store.keyterms("Product / India") == ["Kaushal"]


def test_tts_with_ui_payload_returns_audio_bytes(client, monkeypatch):
    got = {}

    def fake_fetch(voice_id, text):
        got.update(voice_id=voice_id, text=text)
        return b"mp3"

    monkeypatch.setattr("app.api.main._tts_fetch", fake_fetch)
    r = client.post("/tts", json={"text": "Kaushal's tasks."})
    assert r.status_code == 200 and r.content == b"mp3" and r.headers["content-type"] == "audio/mpeg"
    client.post("/tts", json={"text": "hi", "voice": "voice-9"})
    assert got["voice_id"] == "voice-9"


def test_tts_without_api_key_is_a_clean_503(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 503 and "ELEVENLABS_API_KEY" in r.json()["detail"]


# ---------------------------------------------------------------- helpers + serve


def test_shape_helpers():
    assert mmss(65.9) == "01:05"
    assert line_language("कल meeting है", "hin") == "HI + EN"
    assert line_language("नमस्ते", None) == "HI"
    assert line_language("hello", "eng") == "EN"


def test_serve_command_starts_uvicorn_with_the_app(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(app=app, **kw))
    code = cli.main(["serve", "--no-open", "--port", "8123", "--db", str(tmp_path / "x.db")])
    assert code == 0
    assert calls["port"] == 8123 and calls["host"] == "127.0.0.1"
    assert calls["app"].title == "MeetNotes"


def test_ui_source_has_no_secrets_or_mock_data():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "ui" / "src"
    text = "\n".join(p.read_text(encoding="utf-8") for p in src.glob("*.js"))
    assert not re.search(r"xi-api-key|api\.elevenlabs|ELEVENLABS_API_KEY", text)
    assert not re.search(r"\b(mock|fake|sample|demo)", text, re.IGNORECASE)
    json.dumps(text)  # sanity: valid text
