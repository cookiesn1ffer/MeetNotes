import wave
from pathlib import Path

import pytest

from app.capture import AudioSource, FileSource, MicSource, SystemAudioSource


def test_filesource_returns_source_path(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    src = FileSource(audio)
    # `out` is intentionally ignored for a static file source.
    assert src.record(tmp_path / "ignored.wav", 0.0) == audio


def test_filesource_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.wav")


def test_micsource_records_via_injected_callables(tmp_path):
    calls: list[tuple] = []

    def fake_recorder(duration_s: float, sr: int, channels: int) -> bytes:
        calls.append(("rec", duration_s, sr, channels))
        return b"\x00\x00" * int(duration_s * sr * channels)

    def fake_writer(path: Path, pcm: bytes, sr: int, channels: int) -> None:
        calls.append(("wri", path, len(pcm), sr, channels))
        path.write_bytes(pcm)

    out = tmp_path / "sub" / "mic.wav"
    src = MicSource(sample_rate=16000, channels=1, recorder=fake_recorder, writer=fake_writer)
    returned = src.record(out, 1.0)
    assert returned == out
    assert calls == [
        ("rec", 1.0, 16000, 1),
        ("wri", out, 16000 * 2, 16000, 1),
    ]
    assert out.is_file()
    # Parent dir created as needed (cross-platform via pathlib).
    assert out.parent.is_dir()


def test_micsource_default_writer_produces_valid_wav(tmp_path):
    out = tmp_path / "x.wav"

    def rec(duration_s: float, sr: int, channels: int) -> bytes:
        return b"\x00\x00" * int(duration_s * sr * channels)

    MicSource(sample_rate=8000, channels=1, recorder=rec).record(out, 0.5)
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 8000
        assert w.getnchannels() == 1
        assert w.getnframes() == 4000  # 0.5s * 8000Hz


def test_systemaudiosource_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        SystemAudioSource().record(tmp_path / "x.wav", 1.0)


def test_all_sources_subclass_audio_source():
    assert issubclass(FileSource, AudioSource)
    assert issubclass(MicSource, AudioSource)
    assert issubclass(SystemAudioSource, AudioSource)
