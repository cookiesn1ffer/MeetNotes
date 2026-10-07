# MeetNotes

A multilingual (Hindi/English code-switching) meeting summarizer.

## Pipeline

```
audio -> ElevenLabs Scribe -> speaker resolver -> single LLM call -> SQLite
```

- Exactly these stages, in this order. Each stage is its own module under `app/`.
- One LLM call per meeting. Do not split summarization into multiple calls or chains.
- Preserve Hindi/English code-switching in transcripts and summaries. Never translate or transliterate unless explicitly asked. Treat all text as UTF-8 (Devanagari included) everywhere: files, SQLite, logs.

## Hard constraints

- **No agent frameworks, no LangChain, no vector DB.**
- **No new dependencies without asking the user first.** Every dependency in `pyproject.toml` must have a one-line justification as a comment next to it. Prefer the standard library (`sqlite3`, `pathlib`, `logging`, `time`, `argparse`).
- **Windows and Linux must behave identically.**
  - No shell-specific code (no `os.system`, no `subprocess` with `shell=True`, no bash/PowerShell scripts as part of the app).
  - Use `pathlib.Path` for all paths. No hardcoded `/` or `\` separators, no string-concatenated paths.
  - Open text files with explicit `encoding="utf-8"`.
- **Never commit API keys.** Read secrets from `.env` (loaded at startup). Keep `.env` in `.gitignore`; maintain a `.env.example` with names only, no values. Never log or print secret values.

## Testing and observability

- Every module has tests in `tests/` (`tests/test_<module>.py`). New module = new test file in the same change.
- Tests must not call ElevenLabs or the LLM over the network; use fixtures/fakes. Use temporary SQLite databases (`tmp_path`).
- Every pipeline stage emits a timing log (stage name + elapsed seconds, via `time.perf_counter()`) using the `logging` module. `app.bench` reports per-stage timings.

## Commands

```
uv run pytest                  # run tests
uv run ruff check              # lint
uv run python -m app.bench     # benchmark pipeline stage timings
```

Run tests and lint before declaring work done.
