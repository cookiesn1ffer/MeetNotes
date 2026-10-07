from pathlib import Path

import pytest

from app.feedback import FeedbackStore, pipeline_inputs
from app.models import Utterance
from app.speakers import Roster, resolve


@pytest.fixture
def store(tmp_path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "fb.db")


# ---------------------------------------------------------------- glossary


def test_glossary_round_trip_and_keyterms(store):
    store.add_glossary_term("default", "KPI", "acronym")
    store.add_glossary_term("default", "Kaushal", "person")
    assert store.glossary("default") == {"KPI": "acronym", "Kaushal": "person"}
    assert sorted(store.keyterms("default")) == ["KPI", "Kaushal"]


def test_glossary_upsert_updates_kind(store):
    store.add_glossary_term("default", "KPI", "acronym")
    store.add_glossary_term("default", "KPI", "metric")
    assert store.glossary("default") == {"KPI": "metric"}


def test_glossary_scoped_by_team(store):
    store.add_glossary_term("teamA", "foo", "x")
    store.add_glossary_term("teamB", "foo", "y")
    assert store.glossary("teamA") == {"foo": "x"}
    assert store.glossary("teamB") == {"foo": "y"}


# ---------------------------------------------------------------- task corrections / TF-IDF


def test_correct_task_round_trip(store):
    store.correct_task(
        "default", {"task": "wrong"}, {"task": "ppt on planets"}, "ppt on planets"
    )
    out = store.retrieve_examples("default", "ppt on planets")
    assert len(out) == 1
    assert out[0]["before"] == {"task": "wrong"}
    assert out[0]["after"] == {"task": "ppt on planets"}
    assert out[0]["source_quote"] == "ppt on planets"


def test_retrieve_examples_ranks_by_lexical_similarity(store):
    store.correct_task("default", {}, {"task": "ppt"}, "please prepare the ppt on planets")
    store.correct_task("default", {}, {"task": "checklist"}, "launch readiness checklist")
    store.correct_task("default", {}, {"task": "review"}, "schedule the design review")
    top = store.retrieve_examples("default", "we need another ppt on the planets", k=2)
    assert top and top[0]["after"]["task"] == "ppt"


def test_retrieve_examples_returns_up_to_k_and_drops_unrelated(store):
    for i in range(5):
        store.correct_task("default", {}, {"task": f"widget-{i}"}, f"ship widget v{i}")
    top = store.retrieve_examples("default", "ship widget", k=3)
    assert 1 <= len(top) <= 3
    for ex in top:
        assert "widget" in ex["after"]["task"]


def test_retrieve_examples_empty_when_no_corrections(store):
    assert store.retrieve_examples("default", "anything") == []


def test_retrieve_examples_scoped_by_team(store):
    store.correct_task("teamA", {}, {"task": "A"}, "ppt on planets")
    store.correct_task("teamB", {}, {"task": "B"}, "ppt on planets")
    out = store.retrieve_examples("teamA", "ppt on planets")
    assert len(out) == 1 and out[0]["after"]["task"] == "A"


def test_tfidf_handles_devanagari(store):
    store.correct_task("t", {}, {"task": "भेजो"}, "कल तक रिपोर्ट भेजो")
    out = store.retrieve_examples("t", "कल तक रिपोर्ट")
    assert out and out[0]["after"]["task"] == "भेजो"


# ---------------------------------------------------------------- speaker corrections


def test_record_and_fetch_speaker_enrollments(store, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake")
    store.correct_speaker("default", "speaker_0", "Aarush", clip)
    assert store.enrollments_for("default", "Aarush") == [clip]


def test_correct_speaker_without_clip_still_records_name(store):
    store.correct_speaker("default", "speaker_0", "Aarush", None)
    assert store.enrollments_for("default", "Aarush") == []
    roster = store.build_roster("default", {})
    assert "Aarush" in roster.enrollments


def test_build_roster_merges_base_and_corrections(store, tmp_path):
    base_clip = tmp_path / "base.wav"
    base_clip.write_bytes(b"fake")
    corr_clip = tmp_path / "corr.wav"
    corr_clip.write_bytes(b"fake")
    store.correct_speaker("default", "speaker_0", "Aarush", corr_clip)
    roster = store.build_roster("default", {"Aarush": [base_clip]})
    assert roster.enrollments["Aarush"] == [base_clip, corr_clip]


def test_build_roster_adds_new_names(store, tmp_path):
    clip = tmp_path / "p.wav"
    clip.write_bytes(b"fake")
    store.correct_speaker("default", "speaker_1", "Priya", clip)
    roster = store.build_roster("default", {"Aarush": []})
    assert "Priya" in roster.enrollments
    assert roster.enrollments["Priya"] == [clip]


# ---------------------------------------------------------------- end-to-end: better after 3 corrections


def test_resolver_scores_better_after_three_speaker_corrections(store, tmp_path):
    """Same utterances, same embedder: 0/N before, N/N after 3 enrollment corrections."""
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake")
    clips = [tmp_path / f"c{i}.wav" for i in range(3)]
    for p in clips:
        p.write_bytes(b"fake")

    query = [0.9, 0.1]
    enrollment_vectors = {
        clips[0]: [0.6, 0.4],
        clips[1]: [1.0, 0.0],
        clips[2]: [1.1, -0.1],
    }

    def embed_sample(p: Path) -> list[float]:
        return enrollment_vectors[p]

    def embed_turn(a: Path, s: float, e: float) -> list[float]:
        return query

    utterances = [
        Utterance("speaker_0", 0.0, 2.0, "chalo start karte hain", "hin"),
        Utterance("speaker_0", 2.0, 4.0, "aaj ka agenda roadmap hai", "hin"),
    ]

    def accuracy(roster: Roster) -> float:
        out = resolve(
            utterances,
            roster,
            audio=audio,
            threshold=0.9,
            embed_turn=embed_turn,
            embed_sample=embed_sample,
        )
        return sum(1 for u in out if u.speaker == "Aarush") / len(out)

    before = accuracy(Roster({"Aarush": []}))
    for i, clip in enumerate(clips):
        store.correct_speaker("default", "speaker_0", "Aarush", clip)
    after = accuracy(store.build_roster("default", {"Aarush": []}))

    assert before == 0.0
    assert after == 1.0
    assert after > before


# ---------------------------------------------------------------- pipeline_inputs convenience


def test_pipeline_inputs_returns_keyterms_glossary_and_examples(store):
    store.add_glossary_term("default", "KPI", "acronym")
    store.correct_task("default", {}, {"task": "ppt"}, "ppt on planets")
    out = pipeline_inputs(store, "default", "we need another ppt on planets")
    assert out["keyterms"] == ["KPI"]
    assert out["glossary"] == {"KPI": "acronym"}
    assert out["examples"] and out["examples"][0]["after"]["task"] == "ppt"
