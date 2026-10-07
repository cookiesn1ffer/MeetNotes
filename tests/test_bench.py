import json
from pathlib import Path

from app.bench import (
    COLUMNS,
    render_markdown,
    run_bench,
    speaker_accuracy,
    task_metrics,
    wer,
)
from app.models import Utterance

# ---------------------------------------------------------------- metrics


def test_wer_identical_strings_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_counts_substitutions_insertions_deletions():
    # 4 ref words, 1 substitution -> 0.25
    assert wer("the quick brown fox", "the slow brown fox") == 0.25
    # 4 ref words, 1 deletion -> 0.25
    assert wer("the quick brown fox", "the quick fox") == 0.25
    # 4 ref words, 1 insertion -> 0.25
    assert wer("the quick brown fox", "the quick brown red fox") == 0.25


def test_wer_empty_reference_handles_both_sides():
    assert wer("", "") == 0.0
    assert wer("", "hello") == 1.0


def test_speaker_accuracy_time_weighted():
    truth = [
        Utterance("Aarush", 0.0, 2.0, "", "hin"),
        Utterance("Priya", 2.0, 4.0, "", "hin"),
    ]
    # Perfect match: both utterances with the right speaker.
    hyp = [
        Utterance("Aarush", 0.0, 2.0, "", "hin"),
        Utterance("Priya", 2.0, 4.0, "", "hin"),
    ]
    assert speaker_accuracy(hyp, truth) == 1.0
    # Wrong second speaker (2s of 4s total correct).
    hyp2 = [
        Utterance("Aarush", 0.0, 2.0, "", "hin"),
        Utterance("Aarush", 2.0, 4.0, "", "hin"),
    ]
    assert speaker_accuracy(hyp2, truth) == 0.5


def test_task_metrics_fuzzy_text_match():
    truth = [{"assignee": "Kaushal", "task": "ppt on planets"}]
    # Fuzzy variants on the task text with the correct assignee -> matched.
    hyp = [{"assignee": "Kaushal", "task": "ppt about planets"}]
    assert task_metrics(hyp, truth) == (1.0, 1.0)
    # Wrong assignee -> no match.
    hyp2 = [{"assignee": "Aarush", "task": "ppt on planets"}]
    p, r = task_metrics(hyp2, truth)
    assert (p, r) == (0.0, 0.0)


def test_task_metrics_precision_and_recall_on_mix():
    truth = [
        {"assignee": "Kaushal", "task": "ppt on planets"},
        {"assignee": "Priya", "task": "launch checklist"},
    ]
    hyp = [
        {"assignee": "Kaushal", "task": "ppt on planets"},        # TP
        {"assignee": "Aarush", "task": "random thing"},            # FP
    ]
    p, r = task_metrics(hyp, truth)
    assert p == 0.5
    assert r == 0.5


def test_task_metrics_empty_cases():
    assert task_metrics([], []) == (1.0, 1.0)
    assert task_metrics([{"assignee": "A", "task": "x"}], []) == (0.0, 1.0)
    assert task_metrics([], [{"assignee": "A", "task": "x"}]) == (1.0, 0.0)


# ---------------------------------------------------------------- markdown shape


def test_render_markdown_has_all_columns():
    md = render_markdown([])
    for col in COLUMNS:
        assert col in md
    # Threshold line is in the file.
    assert "90%" in md
    assert "80%" in md


# ---------------------------------------------------------------- end-to-end


def _write_truth(path: Path, roster: list[str], task: dict, transcript: str) -> None:
    path.write_text(
        json.dumps(
            {
                "duration_s": 60.0,
                "transcript": transcript,
                "utterances": [
                    {"speaker": "Aarush", "start": 0.0, "end": 60.0, "text": transcript, "lang": "hin"}
                ],
                "roster": roster,
                "tasks": [task],
                "team_id": "demo",
            }
        ),
        encoding="utf-8",
    )


def _scribe_response(transcript: str) -> dict:
    """Build a minimal word-level Scribe response from a transcript string."""
    words = []
    t = 0.0
    for i, w in enumerate(transcript.split()):
        words.append({"type": "word", "text": w, "start": t, "end": t + 0.3, "speaker_id": "speaker_0"})
        if i < len(transcript.split()) - 1:
            words.append({"type": "spacing", "text": " ", "start": t + 0.3, "end": t + 0.4, "speaker_id": "speaker_0"})
        t += 0.4
    return {"language_code": "hin", "words": words}


def test_run_bench_writes_results_md_with_all_columns(tmp_path):
    audio_dir = tmp_path / "eval" / "audio"
    truth_dir = tmp_path / "eval" / "truth"
    audio_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    out = tmp_path / "bench" / "results.md"

    clip = audio_dir / "sample.wav"
    clip.write_bytes(b"audio-bytes")
    transcript = "Kaushal you are assigned to make a ppt on planets for next meeting"
    _write_truth(
        truth_dir / "sample.json",
        roster=["Aarush", "Kaushal"],
        task={"assignee": "Kaushal", "task": "ppt on planets", "deadline": "next meeting"},
        transcript=transcript,
    )

    def fake_post(audio: Path, keyterms) -> dict:
        return _scribe_response(transcript)

    def fake_call(system: str, user: str, schema: dict) -> dict:
        return {
            "summary": "ok",
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

    rows = run_bench(audio_dir, truth_dir, out, post=fake_post, call=fake_call, use_cache=False)

    text = out.read_text(encoding="utf-8")
    for col in COLUMNS:
        assert col in text
    assert "sample.wav" in text
    assert len(rows) == 1
    r = rows[0]
    assert r["clip"] == "sample.wav"
    assert r["duration (s)"] == 60.0
    assert r["WER (%)"] == 0.0  # transcript echoed back
    assert r["task recall (%)"] == 100.0
    assert r["task precision (%)"] == 100.0


def test_run_bench_writes_header_only_when_no_clips(tmp_path):
    audio_dir = tmp_path / "eval" / "audio"
    audio_dir.mkdir(parents=True)
    out = tmp_path / "bench" / "results.md"
    rows = run_bench(audio_dir, tmp_path / "eval" / "truth", out)
    assert rows == []
    text = out.read_text(encoding="utf-8")
    for col in COLUMNS:
        assert col in text


def test_run_bench_skips_audio_without_truth(tmp_path, caplog):
    audio_dir = tmp_path / "eval" / "audio"
    truth_dir = tmp_path / "eval" / "truth"
    audio_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (audio_dir / "orphan.wav").write_bytes(b"a")
    out = tmp_path / "bench" / "results.md"
    rows = run_bench(audio_dir, truth_dir, out)
    assert rows == []
