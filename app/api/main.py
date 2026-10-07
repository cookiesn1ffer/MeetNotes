"""FastAPI backend for the MeetNotes web UI (ui/) and user corrections.

Endpoints:
    POST /process           upload audio -> full pipeline -> JSON for the UI: per-person
                            notes + tasks (`people`), `transcript` lines with speaker/lang,
                            per-stage `stage_timings`, speaker `confidence`, a `run_id`
                            (plus the flat `tasks`/`notes`/`utterances`/`timings` fields)
    POST /tts               proxy to ElevenLabs TTS; returns audio/mpeg (key stays server-side)
    POST /correct_task      write a task correction
    POST /correct_speaker   write a speaker correction; with `run_id` + start/end it also
                            learns the voice from that span of the uploaded audio
    POST /glossary          add or upsert a glossary term
    GET  /                  the built UI (ui/dist) when present, else a minimal fallback page

Errors are JSON `{"detail": "<one line>"}`: 400 bad input, 502 upstream/network, 503 missing
config (e.g. an API key).

Reads of stored corrections/glossary flow through the pipeline (`run_pipeline`), not this API.
"""

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.shape import (
    build_people,
    build_transcript,
    speaker_confidence,
    stage_timings,
    tasks_with_ids,
)
from app.config import get_elevenlabs_key
from app.feedback import FeedbackStore
from app.pipeline import run_pipeline

DEFAULT_TEAM = "default"
DEFAULT_DB = Path("data/meetnotes.db")
STATIC_DIR = Path(__file__).parent / "static"  # minimal fallback page
DEFAULT_UI_DIST = Path(__file__).resolve().parents[2] / "ui" / "dist"
# `pnpm dev` (vite) runs on another localhost port and calls this API cross-origin.
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
TTS_MODEL = "eleven_multilingual_v2"
RACHEL_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs stock voice; override with ELEVENLABS_VOICE_ID


class TaskCorrectionIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    before: dict = Field(default_factory=dict, description="Original (possibly wrong) task.")
    after: dict = Field(..., description="The user's corrected task.")
    source_quote: str = Field(..., description="Verbatim transcript substring the task came from.")


class SpeakerCorrectionIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    before: str = Field(..., description="Original speaker label (diarization label or name).")
    after: str = Field(..., description="Corrected speaker name.")
    clip_path: str | None = Field(None, description="Audit-only path to the enrollment clip.")
    run_id: str | None = Field(None, description="`run_id` from /process; locates the uploaded audio.")
    audio_path: str | None = Field(None, description="Server-side audio path (alternative to run_id).")
    start: float | None = Field(None, description="Start seconds of the utterance span.")
    end: float | None = Field(None, description="End seconds of the utterance span.")
    embedding: list[float] | None = Field(None, description="Precomputed voice embedding.")


class GlossaryTermIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    term: str
    kind: str = Field(..., description="Category of the term (person, product, acronym, ...).")


class TtsIn(BaseModel):
    text: str
    voice: str | None = Field(None, description="ElevenLabs voice id.")
    voice_id: str | None = Field(None, description="Alias of `voice`.")


# ---------------------------------------------------------------- helpers


def _parse_roster(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


def _resolve_voice_id(requested: str | None) -> str:
    return requested or os.environ.get("ELEVENLABS_VOICE_ID") or RACHEL_VOICE_ID


def _tts_fetch(voice_id: str, text: str) -> bytes:
    """ElevenLabs TTS over HTTP; split out so tests can monkeypatch it."""
    body = json.dumps({"text": text, "model_id": TTS_MODEL}).encode("utf-8")
    headers = {
        "xi-api-key": get_elevenlabs_key(),
        "content-type": "application/json",
        "accept": "audio/mpeg",
    }
    req = urllib.request.Request(
        TTS_URL.format(voice_id=voice_id), data=body, method="POST", headers=headers
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _upstream_error(exc: Exception, what: str) -> HTTPException:
    """Map a pipeline/TTS failure to a clean one-line JSON error (never a traceback)."""
    if isinstance(exc, RuntimeError):  # e.g. "ELEVENLABS_API_KEY is not set"
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, urllib.error.HTTPError):
        return HTTPException(status_code=502, detail=f"{what} upstream error {exc.code}: {exc.reason}")
    if isinstance(exc, urllib.error.URLError):
        return HTTPException(status_code=502, detail=f"{what} network failure: {exc.reason}")
    raise exc


def _find_run_audio(uploads: Path, run_id: str) -> Path | None:
    if not RUN_ID_RE.match(run_id):
        return None
    return next(iter(sorted(uploads.glob(f"{run_id}.*"))), None)


# ---------------------------------------------------------------- app


def create_app(store: FeedbackStore | None = None, ui_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI app. Tests pass a store; `ui_dir` is the built UI (default ui/dist)."""
    store = store or FeedbackStore(DEFAULT_DB)
    ui_dir = Path(ui_dir) if ui_dir is not None else DEFAULT_UI_DIST
    app = FastAPI(title="MeetNotes")
    app.state.store = store
    app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN_REGEX,
                       allow_methods=["*"], allow_headers=["*"])

    def get_store() -> FeedbackStore:
        return app.state.store

    def uploads_dir() -> Path:
        return app.state.store.path.parent / "uploads"

    # ------- pipeline

    @app.post("/process")
    async def process(
        audio: UploadFile = File(...),
        team_id: str = Form(DEFAULT_TEAM),
        roster: str = Form(""),
        s: FeedbackStore = Depends(get_store),
    ) -> dict:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty audio upload")
        roster_names = _parse_roster(roster)

        # Kept (not deleted) so a later speaker correction can learn the voice from a span.
        run_id = uuid.uuid4().hex[:12]
        suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(audio.filename or "").suffix)[:8] or ".wav"
        audio_path = uploads_dir() / f"{run_id}{suffix}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(content)
        try:
            result = run_pipeline(audio_path, team_id, roster_names, s)
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            raise _upstream_error(exc, "processing") from exc

        return {
            "run_id": run_id,
            "confidence": speaker_confidence(result),
            "stage_timings": stage_timings(result),
            "people": build_people(result),
            "transcript": build_transcript(result),
            "roster": result.roster,
            "summary": result.summary,
            "notes": result.notes,
            "tasks": tasks_with_ids(result.tasks),
            "utterances": [
                {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text, "lang": u.lang}
                for u in result.utterances
            ],
            "timings": result.stage_logs,
        }

    # ------- TTS proxy

    @app.post("/tts")
    def tts(body: TtsIn) -> Response:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="empty text")
        voice_id = _resolve_voice_id(body.voice or body.voice_id)
        try:
            audio_bytes = _tts_fetch(voice_id, body.text)
        except Exception as exc:
            raise _upstream_error(exc, "text-to-speech") from exc
        return Response(content=audio_bytes, media_type="audio/mpeg")

    # ------- corrections / glossary

    @app.post("/correct_task")
    def correct_task(body: TaskCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        row_id = s.correct_task(body.team_id, body.before, body.after, body.source_quote)
        return {"id": row_id}

    @app.post("/correct_speaker")
    def correct_speaker(body: SpeakerCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        audio = Path(body.audio_path) if body.audio_path else None
        if audio is None and body.run_id:
            audio = _find_run_audio(uploads_dir(), body.run_id)
        embedding = body.embedding
        if embedding is None and audio is not None and body.start is not None and body.end is not None:
            embedding = s.embed_span(audio, body.start, body.end)
        row_id = s.correct_speaker(
            body.team_id,
            body.before,
            body.after,
            Path(body.clip_path) if body.clip_path else None,
            embedding=embedding,
        )
        # False when embeddings aren't installed / the audio can't be decoded: the name
        # correction is still recorded, but no voice was learned.
        return {"id": row_id, "voice_learned": embedding is not None}

    @app.post("/glossary")
    def add_glossary_term(body: GlossaryTermIn, s: FeedbackStore = Depends(get_store)) -> dict:
        s.add_glossary_term(body.team_id, body.term, body.kind)
        return {"ok": True}

    # ------- UI (registered last so API routes take precedence)

    if (ui_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    else:

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

    return app


# Lazy module-level `app` so `uvicorn app.api.main:app` works but importing this
# module (e.g. from tests) does NOT eagerly create a FeedbackStore at DEFAULT_DB.
_lazy_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    if name == "app":
        global _lazy_app
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module 'app.api.main' has no attribute {name!r}")
