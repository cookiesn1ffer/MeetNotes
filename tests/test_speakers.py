from pathlib import Path

from app.models import Utterance
from app.speakers import Roster, _by_transcript, resolve


def u(speaker, start, end, text, lang="hin"):
    return Utterance(speaker, start, end, text, lang)


def test_self_introduction_binds_speaker():
    utts = [
        u("speaker_0", 0, 2, "Hi everyone, I'm Aarush and I'll kick us off"),
        u("speaker_1", 2, 4, "Scope kya hai is quarter ka?"),
    ]
    roster = Roster({"Aarush": [], "Priya": []})
    got = resolve(utts, roster)
    assert got[0].speaker == "Aarush"
    assert got[1].speaker == "Unknown 1"


def test_hindi_self_introduction():
    utts = [u("speaker_2", 0, 2, "मैं Priya हूँ, product side se")]
    roster = Roster({"Priya": []})
    got = resolve(utts, roster)
    assert got[0].speaker == "Priya"


def test_direct_address_binds_next_speaker():
    utts = [
        u("speaker_0", 0, 2, "Priya, kya scope hai is quarter ka?"),
        u("speaker_1", 2, 5, "Scope mostly launch readiness aur onboarding par hoga"),
    ]
    roster = Roster({"Priya": []})
    got = resolve(utts, roster)
    assert got[1].speaker == "Priya"
    assert got[0].speaker == "Unknown 1"


def test_unknown_fallback_numbers_in_order():
    utts = [
        u("speaker_0", 0, 1, "chalo start karte hain"),
        u("speaker_1", 1, 2, "haan"),
        u("speaker_0", 2, 3, "theek hai"),
    ]
    got = resolve(utts, Roster({"Nobody": []}))
    assert [x.speaker for x in got] == ["Unknown 1", "Unknown 2", "Unknown 1"]


def test_longest_name_wins_in_transcript_match():
    utts = [u("speaker_0", 0, 2, "Hi, my name is Priya Sharma")]
    got = resolve(utts, Roster({"Priya": [], "Priya Sharma": []}))
    assert got[0].speaker == "Priya Sharma"


def test_embedding_pass_assigns_above_threshold(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    enroll_aarush = tmp_path / "aarush.wav"
    enroll_aarush.write_bytes(b"fake")
    enroll_priya = tmp_path / "priya.wav"
    enroll_priya.write_bytes(b"fake")

    enrollments = {enroll_aarush: [1.0, 0.0], enroll_priya: [0.0, 1.0]}

    def embed_sample(p: Path) -> list[float]:
        return enrollments[p]

    def embed_turn(a: Path, s: float, e: float) -> list[float]:
        return [0.95, 0.05] if s < 2 else [0.1, 0.9]  # close to Aarush, then Priya

    utts = [
        u("speaker_0", 0, 2, "ok let's begin"),
        u("speaker_1", 2, 4, "sure"),
    ]
    roster = Roster({"Aarush": [enroll_aarush], "Priya": [enroll_priya]})
    got = resolve(
        utts, roster, audio=audio, embed_turn=embed_turn, embed_sample=embed_sample, threshold=0.5
    )
    assert [x.speaker for x in got] == ["Aarush", "Priya"]


def test_embedding_pass_rejects_below_threshold(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    enroll = tmp_path / "a_enroll.wav"
    enroll.write_bytes(b"fake")

    def embed_sample(p: Path) -> list[float]:
        return [1.0, 0.0]

    def embed_turn(a: Path, s: float, e: float) -> list[float]:
        return [0.3, 0.95]  # cosine ~0.3 vs Aarush -> below 0.5 threshold

    utts = [u("speaker_0", 0, 3, "random speech")]
    roster = Roster({"Aarush": [enroll]})
    got = resolve(
        utts, roster, audio=audio, embed_turn=embed_turn, embed_sample=embed_sample
    )
    assert got[0].speaker == "Unknown 1"


def test_embedding_result_still_runs_transcript_for_other_labels(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    enroll = tmp_path / "aarush.wav"
    enroll.write_bytes(b"fake")

    def embed_sample(p: Path) -> list[float]:
        return [1.0, 0.0]

    def embed_turn(a: Path, s: float, e: float) -> list[float]:
        return [1.0, 0.0] if s < 2 else [0.0, 1.0]

    utts = [
        u("speaker_0", 0, 2, "yes team"),
        u("speaker_1", 2, 4, "मैं Priya हूँ"),
    ]
    roster = Roster({"Aarush": [enroll], "Priya": []})
    got = resolve(
        utts, roster, audio=audio, embed_turn=embed_turn, embed_sample=embed_sample
    )
    assert got[0].speaker == "Aarush"
    assert got[1].speaker == "Priya"


def test_turns_shorter_than_min_duration_are_skipped(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    enroll = tmp_path / "aarush.wav"
    enroll.write_bytes(b"fake")

    calls: list[tuple[float, float]] = []

    def embed_sample(p: Path) -> list[float]:
        return [1.0, 0.0]

    def embed_turn(a: Path, s: float, e: float) -> list[float]:
        calls.append((s, e))
        return [1.0, 0.0]

    utts = [
        u("speaker_0", 0.0, 0.2, "hmm"),  # too short, must not be embedded
        u("speaker_0", 1.0, 2.0, "chalo"),
    ]
    resolve(
        utts, Roster({"Aarush": [enroll]}),
        audio=audio, embed_turn=embed_turn, embed_sample=embed_sample,
    )
    assert calls == [(1.0, 2.0)]


def test_by_transcript_is_idempotent():
    utts = [u("speaker_0", 0, 2, "I'm Aarush")]
    first = _by_transcript(utts, ["Aarush"], {})
    second = _by_transcript(utts, ["Aarush"], first)
    assert first == second == {"speaker_0": "Aarush"}
