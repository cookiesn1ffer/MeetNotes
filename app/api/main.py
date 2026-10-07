"""FastAPI endpoints for user corrections.

Keep it thin: the API only writes to the feedback store. Reads flow through
the pipeline (`app.feedback.pipeline_inputs`), not through these endpoints.
"""

from pathlib import Path

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.feedback import FeedbackStore

DEFAULT_TEAM = "default"
DEFAULT_DB = Path("data/meetnotes.db")


class TaskCorrectionIn(BaseModel):
    team: str = DEFAULT_TEAM
    meeting_id: str
    source_quote: str = Field(..., description="Verbatim transcript substring the task came from.")
    correction: dict = Field(..., description="The user's corrected task object.")


class SpeakerCorrectionIn(BaseModel):
    team: str = DEFAULT_TEAM
    meeting_id: str
    raw_label: str = Field(..., description="Original diarization label, e.g. 'speaker_0'.")
    name: str
    clip_path: str | None = Field(
        None, description="Path to an enrollment clip extracted from the meeting audio."
    )


class GlossaryTermIn(BaseModel):
    team: str = DEFAULT_TEAM
    term: str
    definition: str


def create_app(store: FeedbackStore | None = None) -> FastAPI:
    """Build a FastAPI app. Pass a store for tests; otherwise one at DEFAULT_DB is created."""
    store = store or FeedbackStore(DEFAULT_DB)
    app = FastAPI(title="MeetNotes")
    app.state.store = store

    def get_store() -> FeedbackStore:
        return app.state.store

    @app.post("/corrections/task")
    def correct_task(body: TaskCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        row_id = s.record_task_correction(
            body.team, body.meeting_id, body.source_quote, body.correction
        )
        return {"id": row_id}

    @app.post("/corrections/speaker")
    def correct_speaker(body: SpeakerCorrectionIn, s: FeedbackStore = Depends(get_store)) -> dict:
        row_id = s.record_speaker_correction(
            body.team,
            body.meeting_id,
            body.raw_label,
            body.name,
            Path(body.clip_path) if body.clip_path else None,
        )
        return {"id": row_id}

    @app.post("/glossary")
    def add_glossary_term(body: GlossaryTermIn, s: FeedbackStore = Depends(get_store)) -> dict:
        s.add_glossary_term(body.team, body.term, body.definition)
        return {"ok": True}

    return app
