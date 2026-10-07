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
- Prompt 9 — `README.md` (overview, ASCII pipeline, Linux + Windows setup, env
  vars table, bench instructions + columns/thresholds, current table) and
  `DEMO.md` (3-minute script: play clip -> per-person tasks -> correct task +
  speaker -> re-run and show the fix persists -> bench + TTS readout, with
  prep steps and fallbacks).
- Prompt 10 — `app/pipeline.py` with `run_pipeline(audio, team_id, roster, store)
  -> MeetingResult` as the single wiring point for STT (glossary -> keyterms),
  speaker resolution, and extraction (glossary + top-3 TF-IDF few-shot examples
  via `pipeline_inputs`). `/process` and `bench.run_clip` refactored to call it
  (behavior-identical, +1 feedback_lookup log line). CLI entry point via
  `app/__main__.py` + `app/cli.py`: `uv run python -m app process <audio>
  --team T --roster "A,B,C" [--json]` prints per-person tasks + notes + stage
  timings; `--json` emits the full MeetingResult; exits 1 with a one-line error
  (no traceback) on missing file, missing API key, or network failure. Closes
  loose ends "no CLI orchestrator" and "pipeline_inputs has no caller".

- Prompt 11 — Manus UI (`ui/`, Vite, vanilla JS) integrated with the backend. `ui/` was
  flat, so it was restructured into `src/` + `public/` and built to `ui/dist` (committed so
  `uv run python -m app serve` needs no node). Backend: `serve` command (opens the browser;
  `--no-open` to skip), `ui/dist` mounted at `/` (old page stays as fallback), CORS for
  localhost origins, `/tts` accepts `{text, voice?}` and returns audio bytes (key stays
  server-side), clean one-line JSON errors (503 missing key, 502 upstream/network).
  `/process` now also returns `people` (per-person notes + tasks, plus a "Meeting" card),
  `transcript` (speaker, language, mm:ss), `stage_timings`, `confidence`, `run_id`; uploads
  are kept under `data/uploads/<run_id>.*` so `/correct_speaker` (`run_id` + start/end) can
  learn a voice. Extract notes gained an optional `person`. UI: demo/mock mode and fixtures
  removed; `api.js` maps exactly to the backend shapes. 15 contract tests in
  `tests/test_ui_contract.py`.

## In progress
- (none)

## Next
- (none scheduled — open to the next prompt)

## Loose ends
- UI flow not yet clicked through in a real browser (only served + API/JS error paths checked).
- Glossary chips are not reloaded from the backend after a page refresh (no GET /glossary).
- `ui/SKILL.md` and the two VoiceRoom `.md` files from Manus are left untracked.
- CI has been committed but hasn't actually run on this repo yet — first push to
  GitHub will exercise the Ubuntu + Windows matrix.
- `eval/audio/` and `eval/truth/` are empty, so `bench/results.md` and the
  README table are template-only.

## Failing tests
- none (123/123 pass; ruff clean)
