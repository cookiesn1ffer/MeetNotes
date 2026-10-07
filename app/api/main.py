"""FastAPI endpoints for the MeetNotes web UI and user corrections.

Endpoints:
    GET  /                  single-page UI (static/index.html)
    POST /process           upload audio -> full pipeline -> JSON (summary, notes,
                            tasks, utterances, roster, stage timings)
    POST /tts               proxy to ElevenLabs TTS; returns audio/mpeg
    POST /correct_task      write a task correction
    POST /correct_speaker   write a speaker correction (optionally with a voice embedding)
    POST /glossary          add or upsert a glossary term

Reads of stored corrections/glossary flow through the pipeline (`stt.transcribe` and
`extract.extract` auto-merge via team_id+store), not through this API.
"""

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.config import get_elevenlabs_key
from app.extract import extract
from app.feedback import FeedbackStore
from app.speakers import resolve
from app.stt import transcribe

DEFAULT_TEAM = "default"
DEFAULT_DB = Path("data/meetnotes.db")
STATIC_DIR = Path(__file__).parent / "static"

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
    before: str = Field(..., description="Original diarization label, e.g. 'speaker_0'.")
    after: str = Field(..., description="Corrected speaker name.")
    clip_path: str | None = Field(None, description="Audit-only path to the enrollment clip.")
    audio_path: str | None = Field(None, description="Path to the meeting audio for voice-learning.")
    start: float | None = Field(None, description="Start seconds of the utterance span.")
    end: float | None = Field(None, description="End seconds of the utterance span.")
    embedding: list[float] | None = Field(None, description="Precomputed voice embedding.")


class GlossaryTermIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    term: str
    kind: str = Field(..., description="Category of the term (person, product, acronym, ...).")


class TtsIn(BaseModel):
    text: str
    voice_id: str | None = None


# ---------------------------------------------------------------- helpers


def _parse_roster(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


class _StageLogCollector(logging.Handler):
    """Captures `stage=…` log lines from the pipeline modules during a request."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "stage=" in msg:
            self.lines.append(msg)


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


# ---------------------------------------------------------------- app


def create_app(store: FeedbackStore | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a store for tests; otherwise one at DEFAULT_DB is created."""
    store = store or FeedbackStore(DEFAULT_DB)
    app = FastAPI(title="MeetNotes")
    app.state.store = store

    def get_store() -> FeedbackStore:
        return app.state.store

    # ------- UI

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

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

        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        audio_path = Path(tmp)

        collector = _StageLogCollector()
        app_log = logging.getLogger("app")
        prior_level = app_log.level
        app_log.setLevel(logging.INFO)
        app_log.addHandler(collector)
        try:
            audio_path.write_bytes(content)
            utts = transcribe(audio_path, team_id=team_id, store=s)
            # Build a roster from stored embeddings/clips and merge in roster names.
            base = s.build_roster(team_id, {name: [] for name in roster_names})
            resolved = resolve(utts, base)
            result = extract(resolved, roster_names, team_id=team_id, store=s)
        finally:
            app_log.removeHandler(collector)
            app_log.setLevel(prior_level)
            audio_path.unlink(missing_ok=True)

        return {
            "summary": result["summary"],
            "notes": result["notes"],
            "tasks": result["tasks"],
            "utterances": [
                {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text, "lang": u.lang}
                for u in resolved
            ],
            "roster": roster_names,
            "timings": collector.lines,
        }

    # ------- TTS proxy

    @app.post("/tts")
    def tts(body: TtsIn) -> Response:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="empty text")
        voice_id = _resolve_voice_id(body.voice_id)
        try:
            audio_bytes = _tts_fetch(voice_id, body.text)
        except urllib.error.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"TTS upstream {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=f"TTS upstream unreachable: {e.reason}") from e
        return Response(content=audio_bytes, media_type="audio/mpeg")

    # ------- corrections / glossary

    @app.post("/correct_task")
    def correct_task(body: TaskCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        row_id = s.correct_task(body.team_id, body.before, body.after, body.source_quote)
        return {"id": row_id}

    @app.post("/correct_speaker")
    def correct_speaker(body: SpeakerCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        row_id = s.correct_speaker(
            body.team_id,
            body.before,
            body.after,
            Path(body.clip_path) if body.clip_path else None,
            embedding=body.embedding,
            audio=Path(body.audio_path) if body.audio_path else None,
            start=body.start,
            end=body.end,
        )
        return {"id": row_id}

    @app.post("/glossary")
    def add_glossary_term(body: GlossaryTermIn, s: FeedbackStore = Depends(get_store)) -> dict:
        s.add_glossary_term(body.team_id, body.term, body.kind)
        return {"ok": True}

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
