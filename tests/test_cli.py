import json
import urllib.error
from pathlib import Path

import pytest

from app.cli import main
from app.models import Utterance
from app.pipeline import MeetingResult


def _result(**overrides) -> MeetingResult:
    base = {
        "audio_path": Path("clip.wav"),
        "team_id": "demo",
        "roster": ["Aarush", "Kaushal", "Priya"],
        "utterances": [Utterance("Aarush", 0.0, 1.0, "hi team", "eng")],
        "summary": "ok",
        "notes": [{"topic": "Planets", "points": ["Sun ek dwarf star hai"]}],
        "tasks": [
            {
                "assignee": "Kaushal", "assigned_by": "Aarush",
                "task": "ppt on planets", "deadline": "next meeting",
                "source_quote": "...", "confidence": 0.9,
            },
            {
                "assignee": "Priya", "assigned_by": "Aarush",
                "task": "launch checklist", "deadline": None,
                "source_quote": "...", "confidence": 0.8,
            },
        ],
        "stage_seconds": {"stt": 1.0, "speakers": 0.1, "extract": 2.0},
        "stage_logs": [
            "stage=stt elapsed_s=1.000 utterances=12 cached=False",
            "stage=speakers elapsed_s=0.100 labels=3 resolved=3 unknown=0",
            "stage=extract elapsed_s=2.000 tasks=2",
        ],
        "total_seconds": 3.1,
    }
    base.update(overrides)
    return MeetingResult(**base)


def test_cli_process_prints_tasks_per_person_and_timings(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")
    monkeypatch.setattr("app.cli.run_pipeline", lambda *a, **kw: _result())

    code = main(["process", str(audio), "--team", "demo", "--roster", "Aarush, Kaushal, Priya"])
    out = capsys.readouterr().out
    assert code == 0

    # Per-person grouping, deadline rendered when present.
    assert "Kaushal:" in out
    assert "- ppt on planets (next meeting)" in out
    assert "Priya:" in out
    assert "- launch checklist" in out
    # Aarush has no tasks but still appears because it's in the roster.
    assert "Aarush:" in out
    assert "(no tasks)" in out

    # Timings panel from stage_logs.
    assert "stage=stt" in out
    assert "stage=extract" in out
    # Summary + notes.
    assert "Summary:" in out and "ok" in out
    assert "Planets:" in out and "Sun ek dwarf star hai" in out


def test_cli_process_json_output_parses(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")
    monkeypatch.setattr("app.cli.run_pipeline", lambda *a, **kw: _result())

    code = main(["process", str(audio), "--team", "demo", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    parsed = json.loads(out)
    assert parsed["summary"] == "ok"
    assert parsed["tasks"][0]["assignee"] == "Kaushal"
    assert parsed["stage_seconds"]["stt"] == 1.0
    assert parsed["total_seconds"] == 3.1
    assert parsed["audio_path"].endswith("clip.wav")


def test_cli_missing_file_exits_1_with_clean_message(tmp_path, capsys):
    code = main(["process", str(tmp_path / "nope.wav"), "--team", "demo"])
    err = capsys.readouterr().err
    assert code == 1
    assert "error" in err.lower()
    assert "not found" in err.lower()
    assert "Traceback" not in err


def test_cli_missing_api_key_exits_1_with_clean_message(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")

    def fake_run(*a, **kw):
        raise RuntimeError("ELEVENLABS_API_KEY is not set (put it in .env)")

    monkeypatch.setattr("app.cli.run_pipeline", fake_run)
    code = main(["process", str(audio), "--team", "demo"])
    err = capsys.readouterr().err
    assert code == 1
    assert "ELEVENLABS_API_KEY" in err
    assert "Traceback" not in err


def test_cli_network_failure_exits_1_with_clean_message(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"a")

    def fake_run(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("app.cli.run_pipeline", fake_run)
    code = main(["process", str(audio), "--team", "demo"])
    err = capsys.readouterr().err
    assert code == 1
    assert "network" in err.lower()
    assert "connection refused" in err
    assert "Traceback" not in err


def test_cli_requires_subcommand(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    # argparse exits with 2 on usage errors.
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "process" in err or "required" in err.lower()
