"""End-to-end smoke test with mocked STT and LLM. CI runs this as its own step."""

from pathlib import Path

from app.extract import extract
from app.speakers import Roster, resolve
from app.stt import transcribe


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


def test_pipeline_smoke(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-audio-bytes")
    transcript = "Kaushal you are assigned to make a ppt on planets for next meeting"

    def fake_post(audio_path: Path, keyterms) -> dict:
        return _scribe_response(transcript)

    def fake_llm(system: str, user: str, schema: dict) -> dict:
        return {
            "summary": "demo",
            "notes": [],
            "tasks": [
                {
                    "assignee": "Kaushal",
                    "assigned_by": "Aarush",
                    "task": "ppt on planets",
                    "deadline": "next meeting",
                    "source_quote": transcript,
                    "confidence": 0.9,
                }
            ],
        }

    cache_dir = tmp_path / ".cache" / "stt"
    utts = transcribe(audio, cache_dir=cache_dir, post=fake_post)
    assert utts, "Scribe fake produced no utterances"
    assert all(u.text for u in utts)

    resolved = resolve(utts, Roster({"Aarush": [], "Kaushal": []}))
    assert len(resolved) == len(utts)

    result = extract(resolved, roster=["Aarush", "Kaushal"], call=fake_llm)
    assert result["tasks"], "mocked task should survive extract post-processing"
    assert result["tasks"][0]["assignee"] == "Kaushal"
    assert result["tasks"][0]["task"] == "ppt on planets"
    assert result["tasks"][0]["source_quote"] in transcript
