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
        "/corrections/task",
        json={
            "team": "default",
            "meeting_id": "m1",
            "source_quote": "please prep the ppt on planets",
            "correction": {"task": "ppt on planets", "assignee": "Kaushal"},
        },
    )
    assert r.status_code == 200
    assert isinstance(r.json()["id"], int)
    # Round-trip through the store on the same app.
    store = client.app.state.store
    examples = store.top_examples("default", "ppt on planets")
    assert examples[0]["correction"]["task"] == "ppt on planets"


def test_correct_speaker_endpoint_persists(client, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake")
    r = client.post(
        "/corrections/speaker",
        json={
            "meeting_id": "m1",
            "raw_label": "speaker_0",
            "name": "Aarush",
            "clip_path": str(clip),
        },
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.enrollments_for("default", "Aarush") == [Path(clip)]


def test_correct_speaker_endpoint_without_clip(client):
    r = client.post(
        "/corrections/speaker",
        json={"meeting_id": "m1", "raw_label": "speaker_0", "name": "Aarush"},
    )
    assert r.status_code == 200
    store = client.app.state.store
    assert store.enrollments_for("default", "Aarush") == []  # name recorded, no clip


def test_add_glossary_term_endpoint(client):
    r = client.post("/glossary", json={"term": "OKR", "definition": "obj + key results"})
    assert r.status_code == 200
    store = client.app.state.store
    assert store.glossary("default") == {"OKR": "obj + key results"}
