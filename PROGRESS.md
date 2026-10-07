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
- Deferred gaps closed:
  - `extract.py` now dispatches by `LLM_PROVIDER` env (gemini|anthropic, default gemini);
    `_call_gemini` added alongside `_call_claude`
  - `stt.py` now caches Scribe responses under `.cache/stt/<sha256>.json`

## In progress
- (none)

## Next
- 6 bench + pipeline glue
- 7 CI (ubuntu-latest + windows-latest matrix) + audio capture
- 8 UI
- 9 demo script

## Failing tests
- none (59/59 pass; ruff clean)
