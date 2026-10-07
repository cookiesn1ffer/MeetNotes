"""FastAPI endpoints for user corrections and glossary.

Endpoints are thin wrappers over `app.feedback.FeedbackStore`. Reads happen on the
pipeline side (via `pipeline_inputs` or the `team_id`/`store` kwargs on stt/extract),
not through this API.
"""

from pathlib import Path

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.feedback import FeedbackStore

DEFAULT_TEAM = "default"
DEFAULT_DB = Path("data/meetnotes.db")


class TaskCorrectionIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    before: dict = Field(default_factory=dict, description="Original (possibly wrong) task.")
    after: dict = Field(..., description="The user's corrected task.")
    source_quote: str = Field(..., description="Verbatim transcript substring the task came from.")


class SpeakerCorrectionIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    before: str = Field(..., description="Original diarization label, e.g. 'speaker_0'.")
    after: str = Field(..., description="Corrected speaker name.")
    clip_path: str | None = Field(
        None, description="Path to an enrollment clip extracted from the meeting audio."
    )


class GlossaryTermIn(BaseModel):
    team_id: str = DEFAULT_TEAM
    term: str
    kind: str = Field(..., description="Category of the term (person, product, acronym, ...).")


def create_app(store: FeedbackStore | None = None) -> FastAPI:
    """Build a FastAPI app. Pass a store for tests; otherwise one at DEFAULT_DB is created."""
    store = store or FeedbackStore(DEFAULT_DB)
    app = FastAPI(title="MeetNotes")
    app.state.store = store

    def get_store() -> FeedbackStore:
        return app.state.store

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
        )
        return {"id": row_id}

    @app.post("/glossary")
    def add_glossary_term(body: GlossaryTermIn, s: FeedbackStore = Depends(get_store)) -> dict:
        s.add_glossary_term(body.team_id, body.term, body.kind)
        return {"ok": True}

    return app
