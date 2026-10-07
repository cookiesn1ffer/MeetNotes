"""Resolve diarized speaker labels to roster names.

Order per meeting:
1. Embedding pass (optional, needs audio + a voice embedder): mean cosine similarity
   between each raw speaker label's embedding and each enrolled roster embedding.
   Pick the best match when similarity >= threshold.
2. Transcript pass (always runs for still-unresolved labels): self-introductions
   ("my name is X", "I'm X", "मैं X हूँ") and direct address ("X, kya scope hai?").
3. Anything still unresolved becomes "Unknown N", numbered by first appearance.
"""

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.models import Utterance

log = logging.getLogger(__name__)

# Injectable so tests never load torch. Real impl lives in `ecapa_embedder()` below.
Embedder = Callable[[Path, float, float], list[float]]  # (audio, start, end) -> vector
EnrollEmbedder = Callable[[Path], list[float]]  # enrollment clip -> vector

MIN_TURN_S = 0.5  # too short to embed reliably
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Roster:
    """Names to attribute, with optional enrolled audio samples per name."""

    enrollments: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.enrollments.keys())


# ---------------------------------------------------------------- helpers


def _cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else s / (na * nb)


def _weighted_mean(pairs: list[tuple[list[float], float]]) -> list[float]:
    total = sum(w for _, w in pairs)
    dim = len(pairs[0][0])
    return [sum(v[i] * w for v, w in pairs) / total for i in range(dim)]


def _canonical(matched: str, names: list[str]) -> str | None:
    low = matched.lower()
    for n in names:
        if n.lower() == low:
            return n
    return None


# ---------------------------------------------------------------- embedding pass


def _by_embedding(
    utterances: list[Utterance],
    roster: Roster,
    audio: Path,
    embed_turn: Embedder,
    embed_sample: EnrollEmbedder,
    threshold: float,
) -> dict[str, str]:
    enrolled: dict[str, list[float]] = {}
    for name, paths in roster.enrollments.items():
        vecs = [embed_sample(p) for p in paths if p.is_file()]
        if vecs:
            enrolled[name] = _weighted_mean([(v, 1.0) for v in vecs])
    if not enrolled:
        return {}

    turns: dict[str, list[tuple[list[float], float]]] = {}
    for u in utterances:
        dur = u.end - u.start
        if dur < MIN_TURN_S:
            continue
        turns.setdefault(u.speaker, []).append((embed_turn(audio, u.start, u.end), dur))

    out: dict[str, str] = {}
    for raw, pairs in turns.items():
        mean = _weighted_mean(pairs)
        best_name, best_sim = None, -1.0
        for name, nv in enrolled.items():
            sim = _cosine(mean, nv)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= threshold:
            out[raw] = best_name
    return out


# ---------------------------------------------------------------- transcript pass

_INTRO_TEMPLATES = (
    r"\b(?:my\s+name\s+is|i\s*am|i['’]m|this\s+is)\s+{name}\b",
    r"\bmain\s+{name}\s+(?:hoon|hun|hu)\b",
    r"मैं\s+{name}\s+(?:हूँ|हूं|हु)",
)
_ADDRESS_TEMPLATES = (
    r"^\s*{name}\b\s*[,\-:]",
    r"\bhey\s+{name}\b",
    r"\b{name}\s*,\s+(?:what|please|can|could|kya|kaise|batao)",
)


def _compile(templates: tuple[str, ...], names: list[str]) -> list[re.Pattern[str]]:
    # Longest name first so "Priya Sharma" wins over "Priya".
    alts = sorted((re.escape(n) for n in names), key=len, reverse=True)
    name_group = r"(?P<name>" + "|".join(alts) + r")"
    return [re.compile(t.format(name=name_group), re.IGNORECASE) for t in templates]


def _search_any(patterns: list[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return m
    return None


def _by_transcript(
    utterances: list[Utterance],
    names: list[str],
    existing: dict[str, str],
) -> dict[str, str]:
    if not names:
        return dict(existing)
    out = dict(existing)
    taken = set(out.values())
    intros = _compile(_INTRO_TEMPLATES, names)
    addresses = _compile(_ADDRESS_TEMPLATES, names)

    # Pass 1: self-introductions bind the speaker who said them.
    for u in utterances:
        if u.speaker in out:
            continue
        m = _search_any(intros, u.text)
        if m:
            name = _canonical(m.group("name"), names)
            if name and name not in taken:
                out[u.speaker] = name
                taken.add(name)

    # Pass 2: direct address binds the next unresolved speaker who replies.
    pending: str | None = None
    for u in utterances:
        if u.speaker in out:
            pending = None
            continue
        if pending and pending not in taken:
            out[u.speaker] = pending
            taken.add(pending)
            pending = None
            continue
        m = _search_any(addresses, u.text)
        if m:
            pending = _canonical(m.group("name"), names)
    return out


# ---------------------------------------------------------------- public API


def _fill_unknowns(utterances: list[Utterance], resolved: dict[str, str]) -> dict[str, str]:
    out = dict(resolved)
    counter = 1
    for u in utterances:
        if u.speaker not in out:
            out[u.speaker] = f"Unknown {counter}"
            counter += 1
    return out


def resolve(
    utterances: list[Utterance],
    roster: Roster,
    *,
    audio: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    embed_turn: Embedder | None = None,
    embed_sample: EnrollEmbedder | None = None,
) -> list[Utterance]:
    """Return new utterances with `speaker` set to a roster name or 'Unknown N'."""
    t0 = time.perf_counter()
    label_to_name: dict[str, str] = {}
    if (
        audio is not None
        and roster.enrollments
        and embed_turn is not None
        and embed_sample is not None
    ):
        label_to_name = _by_embedding(
            utterances, roster, audio, embed_turn, embed_sample, threshold
        )
    label_to_name = _by_transcript(utterances, roster.names, label_to_name)
    label_to_name = _fill_unknowns(utterances, label_to_name)

    resolved = [
        Utterance(label_to_name[u.speaker], u.start, u.end, u.text, u.lang)
        for u in utterances
    ]
    known = sum(1 for n in label_to_name.values() if not n.startswith("Unknown "))
    unknown = len(label_to_name) - known
    log.info(
        "stage=speakers elapsed_s=%.3f labels=%d resolved=%d unknown=%d",
        time.perf_counter() - t0,
        len(label_to_name),
        known,
        unknown,
    )
    return resolved


# ---------------------------------------------------------------- real ECAPA embedder


def ecapa_embedder(
    cache_dir: Path | None = None,
) -> tuple[Embedder, EnrollEmbedder]:
    """Lazy-loaded ECAPA-TDNN embedder from speechbrain. Raises if the optional deps are missing."""
    try:
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        msg = (
            "ECAPA needs the optional deps: install with `uv sync --extra embeddings`, "
            "or inject your own embedders into resolve()."
        )
        raise RuntimeError(msg) from exc

    savedir = cache_dir or (Path.home() / ".cache" / "meetnotes" / "ecapa")
    savedir.mkdir(parents=True, exist_ok=True)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
    )

    def _vec(wave) -> list[float]:
        emb = classifier.encode_batch(wave).squeeze().detach().cpu().tolist()
        return emb if isinstance(emb, list) else [float(emb)]

    def embed_turn(audio: Path, start: float, end: float) -> list[float]:
        signal, sr = torchaudio.load(str(audio))
        return _vec(signal[:, int(start * sr) : int(end * sr)])

    def embed_sample(audio: Path) -> list[float]:
        signal, _ = torchaudio.load(str(audio))
        return _vec(signal)

    return embed_turn, embed_sample
