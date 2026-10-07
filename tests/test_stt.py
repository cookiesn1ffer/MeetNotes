from pathlib import Path

from app.feedback import FeedbackStore
from app.models import Utterance
from app.stt import to_roman, transcribe, utterances_from_response


def w(text, start, end, speaker="speaker_0", type_="word"):
    return {"text": text, "start": start, "end": end, "type": type_, "speaker_id": speaker}


def sp(start, end, speaker="speaker_0"):
    return w(" ", start, end, speaker, "spacing")


RESPONSE = {
    "language_code": "hin",
    "words": [
        w("कल", 0.0, 0.4),
        sp(0.4, 0.5),
        w("meeting", 0.5, 1.0),
        sp(1.0, 1.1),
        w("है", 1.1, 1.3),
        w("Hello", 2.0, 2.4, "speaker_1"),
        w("(laughter)", 2.4, 2.8, "speaker_1", "audio_event"),
        w("Friday", 9.0, 9.5, "speaker_1"),  # gap > 2s -> new utterance
    ],
}


# ---------------------------------------------------------------- Scribe grouping / Roman normalization


def test_groups_by_speaker_and_gap_and_keeps_both_languages():
    got = utterances_from_response(RESPONSE)
    assert got == [
        Utterance("speaker_0", 0.0, 1.3, "कल meeting है", "hin"),
        Utterance("speaker_1", 2.0, 2.4, "Hello", "hin"),
        Utterance("speaker_1", 9.0, 9.5, "Friday", "hin"),
    ]


def test_roman_option():
    got = utterances_from_response(RESPONSE, roman=True)
    assert got[0].text == "kal meeting hai"


def test_transcribe_uses_injected_post_and_passes_keyterms(tmp_path):
    seen = {}

    def fake_post(audio: Path, keyterms):
        seen["args"] = (audio, list(keyterms))
        return RESPONSE

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    result = transcribe(audio, ["Aarush"], use_cache=False, post=fake_post)
    assert seen["args"] == (audio, ["Aarush"])
    assert len(result) == 3


def test_empty_response():
    assert utterances_from_response({"words": []}) == []


def test_to_roman_hindi_words():
    assert to_roman("नमस्ते") == "namaste"
    assert to_roman("कल") == "kal"
    assert to_roman("समझ") == "samajh"
    assert to_roman("मैं") == "main"
    assert to_roman("कहाँ") == "kahaan"
    assert to_roman("ज़रूर") == "zaroor"
    assert to_roman("क्या") == "kya"


def test_to_roman_leaves_latin_and_mixed_text():
    assert to_roman("Please send the report") == "Please send the report"
    assert to_roman("कल meeting है।") == "kal meeting hai."
    assert to_roman("") == ""
    assert to_roman("२०२५") == "2025"


# ---------------------------------------------------------------- sha256 cache


def test_cache_hit_returns_same_result_and_makes_zero_calls(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    cache_dir = tmp_path / ".cache" / "stt"
    calls: list[int] = []

    def fake_post(a: Path, kw):
        calls.append(1)
        return RESPONSE

    first = transcribe(audio, cache_dir=cache_dir, post=fake_post)
    second = transcribe(audio, cache_dir=cache_dir, post=fake_post)
    assert len(calls) == 1
    assert first == second


def test_cache_disabled_calls_scribe_each_time(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    calls: list[int] = []

    def fake_post(a: Path, kw):
        calls.append(1)
        return RESPONSE

    transcribe(audio, use_cache=False, post=fake_post)
    transcribe(audio, use_cache=False, post=fake_post)
    assert len(calls) == 2


def test_cache_distinguishes_files_by_content_hash(tmp_path):
    cache_dir = tmp_path / ".cache" / "stt"
    a = tmp_path / "a.wav"
    a.write_bytes(b"aaa")
    b = tmp_path / "b.wav"
    b.write_bytes(b"bbb")
    calls: list[Path] = []

    def fake_post(audio: Path, kw):
        calls.append(audio)
        return RESPONSE

    transcribe(a, cache_dir=cache_dir, post=fake_post)
    transcribe(b, cache_dir=cache_dir, post=fake_post)
    transcribe(a, cache_dir=cache_dir, post=fake_post)
    assert calls == [a, b]


# ---------------------------------------------------------------- team glossary -> keyterms


def test_glossary_term_appears_in_next_stt_call_for_team(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("teamA", "Kaushal", "person")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"xy")
    received: list[list[str]] = []

    def fake_post(a: Path, kw):
        received.append(list(kw))
        return RESPONSE

    transcribe(audio, team_id="teamA", store=store, use_cache=False, post=fake_post)
    assert received == [["Kaushal"]]


def test_glossary_term_not_shared_across_teams(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("teamA", "Kaushal", "person")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"xy")
    received: list[list[str]] = []

    def fake_post(a: Path, kw):
        received.append(list(kw))
        return RESPONSE

    transcribe(audio, team_id="teamB", store=store, use_cache=False, post=fake_post)
    assert received == [[]]


def test_explicit_keyterms_merge_with_team_glossary(tmp_path):
    store = FeedbackStore(tmp_path / "fb.db")
    store.add_glossary_term("teamA", "Kaushal", "person")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"xy")
    received: list[list[str]] = []

    def fake_post(a: Path, kw):
        received.append(list(kw))
        return RESPONSE

    transcribe(
        audio,
        ["Aarush"],
        team_id="teamA",
        store=store,
        use_cache=False,
        post=fake_post,
    )
    assert sorted(received[0]) == ["Aarush", "Kaushal"]
