# PROGRESS

## Done
- Prompt 0 — AGENTS.md + CLAUDE.md
- Prompt 1 — repo scaffold, data models
- Prompt 2 — `app/stt.py` (ElevenLabs Scribe, diarization, word timestamps, Roman normalization) + `scripts/hinglish_check.py`
- Prompt 3 — `app/speakers.py` (ECAPA + transcript fallback) + `scripts/speaker_eval.py`
- Prompt 4 — `app/extract.py` (single LLM call, tool-use/response_schema JSON, fuzzy assignee match)
- Prompt 5 — `app/feedback.py` + `app/api/main.py` (glossary, corrections, TF-IDF retrieval, roster merge)
- Prompt 5a-c — schema reset (`corrections` + `glossary` tables), renamed endpoints
  (`/correct_task`, `/correct_speaker`, `/glossary`); glossary wired into `stt.py` as keyterms
  and into `extract.py`'s prompt per `team_id`
- Prompt 5d — `FeedbackStore.retrieve_examples(team_id, transcript, k=3)` using sklearn
  `TfidfVectorizer` + `cosine_similarity`; `extract.py` prompt now formats few-shot
  examples as `wrong output -> corrected output`
- Prompt 5e — `correct_speaker` now computes an ECAPA embedding from the utterance audio
  span (speechbrain, CPU) and folds it into a running mean per `(team_id, name)` stored
  in a new `speaker_embeddings` table; `Roster.embeddings` carries precomputed means and
  wins over file-based enrollments in `_by_embedding`; speaker-match threshold now reads
  `VOICE_MATCH_THRESHOLD` from config; `scripts/learning_demo.py` runs the before/after
  comparison and prints both numbers
- Deferred gaps closed:
  - `extract.py` now dispatches by `LLM_PROVIDER` env (gemini|anthropic, default gemini);
    `_call_gemini` added alongside `_call_claude`
  - `stt.py` now caches Scribe responses under `.cache/stt/<sha256>.json`

- Prompt 6 — `app/bench.py` (`uv run python -m app.bench`): per-clip metrics
  (time-to-notes, STT time, LLM time, WER, time-weighted speaker accuracy,
  task precision + recall with assignee-exact + fuzzy-text matching) written to
  `bench/results.md`; PASS thresholds documented in the module header
- Prompt 7 — `.github/workflows/ci.yml` (ubuntu-latest + windows-latest matrix,
  `uv sync` + ruff + pytest + smoke test); `app/capture.py` with the AudioSource
  interface (`FileSource`, `MicSource` with lazy sounddevice, `SystemAudioSource`
  stub carrying Windows WASAPI loopback / Linux PipeWire monitor TODOs);
  `tests/test_smoke_pipeline.py` for the mocked end-to-end run
- Prompt 8 — single-page web UI served by FastAPI (`app/api/static/index.html`,
  plain HTML + vanilla JS, no framework): upload or MediaRecorder record, roster
  input, per-person task list with inline edit (POSTs `/correct_task`), notes,
  transcript, timings panel from captured `stage=…` log lines, and a "read my
  tasks" ElevenLabs TTS proxy at `/tts`. New `/process` endpoint runs the full
  pipeline on an uploaded clip (team-scoped glossary + corrections auto-wired)
  and returns summary/notes/tasks/utterances/roster/timings as JSON.

## In progress
- (none)

## Next
- 9 demo script

## Failing tests
- none (97/97 pass; ruff clean)
