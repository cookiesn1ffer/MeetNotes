"""Audio capture sources for the pipeline.

The pipeline consumes a Path to an audio file. An AudioSource knows how to produce
one, whether from an existing file or a live stream. Three concrete sources:

    FileSource         wraps an existing audio file on disk
    MicSource          records from the default microphone (sounddevice, lazy-imported)
    SystemAudioSource  captures the system audio output; NOT YET IMPLEMENTED
                       - Windows: WASAPI loopback via sounddevice extra_settings
                       - Linux:   PipeWire sink monitor (pw-record --target=<sink>.monitor)
                                  or PulseAudio parec --device=<sink>.monitor
                       - macOS:   needs a virtual device like BlackHole

sounddevice is lazy-imported inside MicSource so this module imports cleanly on
CI runners without PortAudio installed. Tests inject the `recorder` and `writer`
callables to stay fully offline.
"""

import logging
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

Recorder = Callable[[float, int, int], bytes]  # (duration_s, sr, channels) -> int16 PCM
Writer = Callable[[Path, bytes, int, int], None]  # (path, pcm, sr, channels)


# ---------------------------------------------------------------- interface


class AudioSource(ABC):
    """Produces an audio file. `record` always returns the Path containing the audio."""

    @abstractmethod
    def record(self, out: Path, duration_s: float) -> Path:
        """For a static file, duration_s is ignored and the source path may be
        returned instead of `out`. For live sources, record up to `duration_s`
        seconds into `out` (WAV).
        """


# ---------------------------------------------------------------- FileSource


class FileSource(AudioSource):
    """A pre-existing audio file on disk. `record` returns the source path as-is."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            msg = f"audio file not found: {self.path}"
            raise FileNotFoundError(msg)

    def record(self, out: Path, duration_s: float = 0.0) -> Path:
        return self.path


# ---------------------------------------------------------------- MicSource helpers


def _default_recorder(duration_s: float, sr: int, channels: int) -> bytes:
    """Record from the default microphone using sounddevice; lazy-imported."""
    import sounddevice as sd

    frames = round(duration_s * sr)
    data = sd.rec(frames, samplerate=sr, channels=channels, dtype="int16")
    sd.wait()
    return bytes(data)


def _default_writer(path: Path, pcm: bytes, sr: int, channels: int) -> None:
    """Write int16 PCM bytes to a WAV using the stdlib `wave` module."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # int16
        w.setframerate(sr)
        w.writeframes(pcm)


# ---------------------------------------------------------------- MicSource


class MicSource(AudioSource):
    """Record from the default microphone to a WAV file.

    `recorder` and `writer` are injectable so tests don't need PortAudio installed.
    Install the optional dep for live use: `uv sync --extra audio`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        *,
        recorder: Recorder | None = None,
        writer: Writer | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._recorder = recorder or _default_recorder
        self._writer = writer or _default_writer

    def record(self, out: Path, duration_s: float) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pcm = self._recorder(duration_s, self.sample_rate, self.channels)
        self._writer(out, pcm, self.sample_rate, self.channels)
        log.info(
            "stage=capture source=mic duration_s=%.2f sr=%d channels=%d out=%s",
            duration_s,
            self.sample_rate,
            self.channels,
            out,
        )
        return out


# ---------------------------------------------------------------- SystemAudioSource


class SystemAudioSource(AudioSource):
    """Capture system audio output (loopback). NOT YET IMPLEMENTED.

    Implementation notes to pick up when this moves forward:

      Windows: sounddevice with WASAPI loopback -
          settings = sd.WasapiSettings(exclusive=False)
          sd.InputStream(..., extra_settings=settings, device=<default output device>)
          Needs the output device index + WASAPI loopback flag.

      Linux (PipeWire): record from the default sink's monitor source -
          `pw-record --target=<sink>.monitor out.wav`
          Or enumerate `.monitor` input devices via sounddevice and pick the one
          matching the current default sink.

      Linux (PulseAudio only): parec --device=<sink>.monitor out.raw
          (then pack with sox / wave).

      macOS: no native loopback; install BlackHole / Loopback as a virtual device
          and record from it like any other input.
    """

    def record(self, out: Path, duration_s: float) -> Path:
        msg = (
            "SystemAudioSource is not yet implemented. "
            "See the class docstring for Windows / Linux / macOS implementation notes."
        )
        raise NotImplementedError(msg)
