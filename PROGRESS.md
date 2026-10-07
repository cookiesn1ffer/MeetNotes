# PROGRESS

## Done
- Prompt 0 — AGENTS.md + CLAUDE.md
- Prompt 1 — repo scaffold, data models
- Prompt 2 — `app/stt.py` (ElevenLabs Scribe, diarization, word timestamps, Roman normalization) + `scripts/hinglish_check.py`
- Prompt 3 — `app/speakers.py` (ECAPA + transcript fallback) + `scripts/speaker_eval.py`
- Prompt 4 — `app/extract.py` (single Claude call, tool-use JSON, fuzzy assignee match)
- Prompt 5 — `app/feedback.py` + `app/api/main.py` (glossary, task + speaker corrections, TF-IDF retrieval, roster merge)

## In progress
- (none)

## Next
- 6 bench + pipeline glue (also wires `feedback.pipeline_inputs` into stt+extract)
- 7 CI (ubuntu-latest + windows-latest matrix) + audio capture
- 8 UI
- 9 demo script

## Deferred / known gaps
- `app/extract.py` is Anthropic-only; env rule says `LLM_PROVIDER` switch, default `gemini`. Fix on the next touch of extract.py.
- `app/stt.py` has no cache; env rule says hash-based cache under `.cache/stt/`. Fix on the next touch of stt.py.
- Repo is not a git repository yet, so the "commit after every task" step is skipped until `git init` is run.

## Failing tests
- none (43/43 pass; ruff clean)
