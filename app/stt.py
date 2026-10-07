"""audio -> list[Utterance] via ElevenLabs Scribe (diarized, word timestamps, auto language)."""

import hashlib
import json
import logging
import mimetypes
import time
import unicodedata
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import get_elevenlabs_key
from app.models import Utterance

if TYPE_CHECKING:
    from app.feedback import FeedbackStore

log = logging.getLogger(__name__)

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL = "scribe_v1"
GAP_SPLIT_S = 2.0  # same speaker, silence longer than this starts a new utterance
DEFAULT_CACHE_DIR = Path(".cache") / "stt"

# ---------------------------------------------------------------- Roman normalization

_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}  # fmt: skip
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}  # fmt: skip
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}  # fmt: skip
_NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_DIGITS = {chr(0x966 + i): str(i) for i in range(10)}
_VIRAMA = "्"
_NUKTA = "़"


def _is_devanagari(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ"


def _roman_word(word: str) -> str:
    """Transliterate one Devanagari run to Hinglish-style Roman with word-final schwa deletion."""
    chars = list(unicodedata.normalize("NFD", word))  # split precomposed nukta letters
    out: list[str] = []
    i = 0
    post_virama = False  # the current consonant forms the tail of a conjunct
    while i < len(chars):
        ch = chars[i]
        if ch in _CONSONANTS:
            nukta = i + 1 < len(chars) and chars[i + 1] == _NUKTA
            out.append(
                _NUKTA_CONSONANTS[ch]
                if nukta and ch in _NUKTA_CONSONANTS
                else _CONSONANTS[ch]
            )
            i += 2 if nukta else 1
            nxt = chars[i] if i < len(chars) else ""
            # After a conjunct (…+virama+C), a word-final 'ा' is written as 'a' (क्या -> kya),
            # not 'aa' (काम stays kaam). This matches common Hinglish usage.
            if post_virama and nxt == "ा":
                out.append("a")
                i += 1
            elif nxt in _CONSONANTS or nxt in _INDEPENDENT_VOWELS:
                out.append("a")  # inherent schwa before another consonant
            elif nxt in _SIGNS and i + 1 < len(chars):
                out.append("a")  # inherent schwa before a non-final anusvara/visarga
            post_virama = False
            continue
        if ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch == _VIRAMA:
            post_virama = True
        elif ch == _NUKTA:
            pass
        elif ch == "।":
            out.append(".")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def to_roman(text: str) -> str:
    """Normalize mixed Devanagari/English text to Roman (Hinglish). Latin text is untouched."""
    result: list[str] = []
    run: list[str] = []
    for ch in text:
        if _is_devanagari(ch):
            run.append(ch)
        else:
            if run:
                result.append(_roman_word("".join(run)))
                run = []
            result.append(ch)
    if run:
        result.append(_roman_word("".join(run)))
    return "".join(result)


# ---------------------------------------------------------------- Scribe client


def _multipart(fields: list[tuple[str, str]], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post_scribe(audio: Path, keyterms: list[str] | None) -> dict:
    fields = [
        ("model_id", SCRIBE_MODEL),
        ("diarize", "true"),
        ("timestamps_granularity", "word"),
        ("tag_audio_events", "false"),
    ]  # no language_code -> Scribe auto-detects
    fields += [("keyterms", term) for term in keyterms or []]
    body, ctype = _multipart(fields, "file", audio)
    req = urllib.request.Request(
        SCRIBE_URL,
        data=body,
        method="POST",
        headers={"xi-api-key": get_elevenlabs_key(), "Content-Type": ctype},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def utterances_from_response(data: dict, *, roman: bool = False) -> list[Utterance]:
    """Group Scribe word-level output into one Utterance per speaker turn."""
    lang = data.get("language_code")
    utterances: list[Utterance] = []
    cur_speaker: str | None = None
    cur_words: list[str] = []
    cur_start = cur_end = 0.0

    def flush() -> None:
        if cur_speaker is None:
            return
        text = "".join(cur_words).strip()
        if text:
            utterances.append(
                Utterance(
                    cur_speaker,
                    cur_start,
                    cur_end,
                    to_roman(text) if roman else text,
                    lang,
                )
            )

    for w in data.get("words", []):
        if w.get("type") == "audio_event":
            continue
        speaker = w.get("speaker_id") or cur_speaker or "speaker_0"
        is_spacing = w.get("type") == "spacing"
        if not is_spacing:
            gap = w["start"] - cur_end if cur_words else 0.0
            if speaker != cur_speaker or gap > GAP_SPLIT_S:
                flush()
                cur_speaker, cur_words, cur_start = speaker, [], w["start"]
            cur_end = w["end"]
        if cur_speaker is not None:
            cur_words.append(w["text"])
    flush()
    return utterances


def _audio_sha256(audio: Path) -> str:
    h = hashlib.sha256()
    with audio.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_file(audio: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{_audio_sha256(audio)}.json"


def transcribe(
    audio: Path,
    keyterms: list[str] | None = None,
    *,
    team_id: str | None = None,
    store: "FeedbackStore | None" = None,
    roman: bool = False,
    post: Callable[[Path, list[str] | None], dict] = _post_scribe,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> list[Utterance]:
    """Transcribe `audio` with diarization and word timestamps; language is auto-detected.

    If `team_id` and `store` are given, glossary terms for that team are merged into
    `keyterms` before calling Scribe. If `use_cache` is set (the default), the raw
    Scribe response is cached under `.cache/stt/<sha256>.json`; a second call for the
    same audio file returns the cached response without hitting ElevenLabs.
    """
    t0 = time.perf_counter()
    audio = Path(audio)

    merged_keyterms = list(keyterms or [])
    if team_id and store is not None:
        for term in store.keyterms(team_id):
            if term not in merged_keyterms:
                merged_keyterms.append(term)

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = _cache_file(audio, cache_dir) if use_cache else None

    if cache_path is not None and cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = True
    else:
        data = post(audio, merged_keyterms)
        cached = False
        if cache_path is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = utterances_from_response(data, roman=roman)
    log.info(
        "stage=stt elapsed_s=%.3f utterances=%d cached=%s",
        time.perf_counter() - t0,
        len(result),
        cached,
    )
    return result
