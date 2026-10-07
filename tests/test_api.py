from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.feedback import FeedbackStore


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
