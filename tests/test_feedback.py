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
    store.add_glossary_term("default", "KPI", "key performance indicator")
    store.add_glossary_term("default", "OKR", "objectives and key results")
    assert store.glossary("default") == {
        "KPI": "key performance indicator",
        "OKR": "objectives and key results",
    }
    assert sorted(store.keyterms("default")) == ["KPI", "OKR"]


def test_glossary_upsert_updates_definition(store):
    store.add_glossary_term("default", "KPI", "old")
    store.add_glossary_term("default", "KPI", "new")
    assert store.glossary("default") == {"KPI": "new"}


def test_glossary_scoped_by_team(store):
    store.add_glossary_term("teamA", "foo", "A")
    store.add_glossary_term("teamB", "foo", "B")
    assert store.glossary("teamA") == {"foo": "A"}
    assert store.glossary("teamB") == {"foo": "B"}


# ---------------------------------------------------------------- task corrections / TF-IDF


def test_top_examples_ranks_by_lexical_similarity(store):
    store.record_task_correction(
        "default", "m1", "please prepare the ppt on planets", {"task": "ppt on planets"}
    )
    store.record_task_correction(
        "default", "m1", "launch readiness checklist", {"task": "launch checklist"}
    )
    store.record_task_correction(
        "default", "m1", "schedule the design review", {"task": "design review"}
    )
    top = store.top_examples("default", "we need another ppt on the planets", k=2)
    assert len(top) == 1 or top[0]["correction"]["task"] == "ppt on planets"
    # Request top=2: either only the overlapping one is returned (strict), or it ranks first.
    assert top[0]["correction"]["task"] == "ppt on planets"


def test_top_examples_returns_up_to_k_and_drops_unrelated(store):
    for i in range(5):
        store.record_task_correction(
            "default", f"m{i}", f"ship widget v{i}", {"task": f"widget-{i}"}
        )
    top = store.top_examples("default", "ship widget", k=3)
    assert 1 <= len(top) <= 3
    for ex in top:
        assert "widget" in ex["correction"]["task"]


def test_top_examples_empty_when_no_corrections(store):
    assert store.top_examples("default", "anything") == []


def test_top_examples_scoped_by_team(store):
    store.record_task_correction("teamA", "m1", "ppt on planets", {"task": "A"})
    store.record_task_correction("teamB", "m1", "ppt on planets", {"task": "B"})
    out = store.top_examples("teamA", "ppt on planets")
    assert len(out) == 1 and out[0]["correction"]["task"] == "A"


def test_tfidf_handles_devanagari(store):
    store.record_task_correction("t", "m", "कल तक रिपोर्ट भेजो", {"task": "भेजो"})
    out = store.top_examples("t", "कल तक रिपोर्ट")
    assert out and out[0]["correction"]["task"] == "भेजो"


# ---------------------------------------------------------------- speaker corrections


def test_record_and_fetch_speaker_enrollments(store, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake")
    store.record_speaker_correction("default", "m1", "speaker_0", "Aarush", clip)
    assert store.enrollments_for("default", "Aarush") == [clip]


def test_build_roster_merges_base_and_corrections(store, tmp_path):
    base_clip = tmp_path / "base.wav"
    base_clip.write_bytes(b"fake")
    corr_clip = tmp_path / "corr.wav"
    corr_clip.write_bytes(b"fake")
    store.record_speaker_correction("default", "m1", "speaker_0", "Aarush", corr_clip)
    roster = store.build_roster("default", {"Aarush": [base_clip]})
    assert roster.enrollments["Aarush"] == [base_clip, corr_clip]


def test_build_roster_adds_new_names(store, tmp_path):
    clip = tmp_path / "p.wav"
    clip.write_bytes(b"fake")
    store.record_speaker_correction("default", "m1", "speaker_1", "Priya", clip)
    roster = store.build_roster("default", {"Aarush": []})
    assert "Priya" in roster.enrollments
    assert roster.enrollments["Priya"] == [clip]


# ---------------------------------------------------------------- end-to-end: resolver scores better


def test_resolver_scores_better_after_three_speaker_corrections(store, tmp_path):
    """Same eval clip, same resolver call: 0/N correct before, N/N after 3 corrections."""
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake")
    clips = [tmp_path / f"c{i}.wav" for i in range(3)]
    for p in clips:
        p.write_bytes(b"fake")

    # Fake embeddings: query vector plus three enrollments that AVERAGE to the query
    # but individually fall below the 0.9 cosine threshold.
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
        store.record_speaker_correction("default", f"m{i}", "speaker_0", "Aarush", clip)

    after = accuracy(store.build_roster("default", {"Aarush": []}))

    assert before == 0.0, "no enrollments -> no match"
    assert after == 1.0, "three corrections -> every utterance resolves"
    assert after > before


# ---------------------------------------------------------------- pipeline_inputs shortcut


def test_pipeline_inputs_returns_keyterms_glossary_and_examples(store):
    store.add_glossary_term("default", "KPI", "key perf indicator")
    store.record_task_correction("default", "m1", "ppt on planets", {"task": "ppt"})
    out = pipeline_inputs(store, "default", "we need another ppt on planets")
    assert out["keyterms"] == ["KPI"]
    assert out["glossary"] == {"KPI": "key perf indicator"}
    assert out["examples"] and out["examples"][0]["correction"]["task"] == "ppt"
