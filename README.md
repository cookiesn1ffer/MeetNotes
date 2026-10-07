# MeetNotes

A multilingual (Hindi / English code-switching) meeting summarizer.

Audio in, per-person tasks and notes out. Preserves Hinglish code-switching
end-to-end. Learns from your corrections — fix a task or a speaker name once,
and the next run of the same meeting (or a similar one for the same team)
reflects the fix via a stored embedding or a few-shot example.

## Pipeline

```
                                  team_id ─┐
                                           ▼
┌──────────┐   ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐   ┌──────────┐
│  audio   │──▶│  ElevenLabs Scribe  │──▶│  speaker resolver   │──▶│  single LLM call │──▶│  SQLite  │
│  file /  │   │  diarization +      │   │  ECAPA embeddings   │   │  Gemini /        │   │ tasks,   │
│  mic /   │   │  word timestamps    │   │  + transcript cues  │   │  Anthropic       │   │ notes,   │
│  upload  │   │  (sha256 cache)     │   │  (running means)    │   │  (tool-use JSON) │   │ feedback │
└──────────┘   └─────────────────────┘   └─────────────────────┘   └──────────────────┘   └──────────┘
                        ▲                          ▲                         ▲                 │
                        │ keyterms                 │ Roster.embeddings       │ few-shot        │
                        └──────────────────────────┴─────────────────────────┤ examples        │
                                                                             │ (TF-IDF top-3)  │
                                                                             └─────────────────┘

Every stage emits a `stage=<name> elapsed_s=<n>` log line.
`app/bench.py` scores a full run against ground truth and writes a markdown table.
```

Exactly these four stages, in this order, with no agent framework and no vector
DB. One LLM call per meeting.

## Setup

The project ships `pyproject.toml` for [uv](https://docs.astral.sh/uv/).
Python 3.11+ (CI pins 3.12).

### Linux (Omarchy / Arch / Debian / Ubuntu / Fedora)

```sh
# Install uv (one line, no sudo):
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root:
uv sync                          # core deps
uv sync --extra embeddings       # optional: ECAPA voice ID (speechbrain + torchaudio, CPU)
uv sync --extra audio            # optional: MicSource (sounddevice; needs libportaudio2)

# Install PyTorch from the CPU index (don't ever pull a CUDA wheel):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env             # then fill in keys
uv run pytest                    # 97 tests
uv run ruff check                # lint
```

For `sounddevice`-backed mic capture, make sure PortAudio is present:
`sudo pacman -S portaudio` on Arch/Omarchy, `sudo apt install libportaudio2`
on Debian/Ubuntu.

### Windows

```powershell
# Install uv (one line, PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# From the repo root:
uv sync
uv sync --extra embeddings
uv sync --extra audio
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

copy .env.example .env           # then fill in keys with Notepad or VS Code
uv run pytest
uv run ruff check
```

PortAudio ships with the `sounddevice` wheel on Windows — no extra install.

### Run the web UI

```sh
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in Firefox or Chromium. The UI is a single-page
HTML (no framework): upload or record audio, enter the roster, click **Process**,
edit tasks inline (the correction is stored), hit **Process** again on the same
clip to see the fix applied. "🔊 read my tasks" plays an ElevenLabs TTS readout
of a given person's tasks.

## Env vars

All read from `.env` at the repo root (never commit it). `.env.example`:

| var                     | required for               | notes                                          |
| ----------------------- | -------------------------- | ---------------------------------------------- |
| `ELEVENLABS_API_KEY`    | STT and TTS                | ElevenLabs Scribe + text-to-speech             |
| `GEMINI_API_KEY`        | extract (default provider) | Google Generative Language API                 |
| `ANTHROPIC_API_KEY`     | extract (alt provider)     | used when `LLM_PROVIDER=anthropic`             |
| `LLM_PROVIDER`          | extract                    | `gemini` (default) or `anthropic`              |
| `VOICE_MATCH_THRESHOLD` | speaker resolve            | cosine threshold for ECAPA match (default 0.5) |
| `ELEVENLABS_VOICE_ID`   | TTS                        | optional; defaults to Rachel                   |

## How to run the benchmark

Drop clips and matching ground truth under `eval/`:

```
eval/
├── audio/
│   └── <id>.wav          # or mp3/m4a/flac/ogg/opus/mp4/webm
└── truth/
    └── <id>.json
```

Each `truth/<id>.json` carries:

```json
{
  "duration_s": 60.0,
  "transcript": "<reference transcript as a single string>",
  "utterances": [
    {"speaker": "Aarush", "start": 0.0, "end": 2.0, "text": "...", "lang": "hin"}
  ],
  "roster": ["Aarush", "Kaushal"],
  "tasks": [{"assignee": "Kaushal", "task": "ppt on planets", "deadline": "next meeting"}],
  "team_id": "demo"
}
```

Then:

```sh
uv run python -m app.bench
# -> Wrote bench/results.md (N clip(s))
```

### PASS thresholds (per clip)

- time-to-notes ≤ 0.25 × duration (so < 15 s on a 60 s clip)
- speaker accuracy ≥ 90 %
- task recall ≥ 80 %

### Current benchmark table

Last run on an empty eval set — drop clips under `eval/audio/` + ground truth under
`eval/truth/` and re-run to replace this table with real numbers. The columns and
threshold footer are fixed so your results slot straight in.

| clip | duration (s) | time-to-notes (s) | STT time (s) | LLM time (s) | WER (%) | speaker acc (%) | task precision (%) | task recall (%) |
| ---- | ------------ | ----------------- | ------------ | ------------ | ------- | --------------- | ------------------ | --------------- |
| _(no clips yet)_ |

_PASS thresholds: time-to-notes ≤ 0.25 × duration, speaker acc ≥ 90 %, task recall ≥ 80 %._

## Commands

```sh
uv run pytest                    # tests
uv run ruff check                # lint
uv run python -m app.bench       # benchmark
uv run python -m app serve       # web UI + API (http://127.0.0.1:8000)

# scripts
uv run python scripts/hinglish_check.py   # eval clips: PASS when a mixed-script transcript keeps both languages
uv run python scripts/speaker_eval.py     # speaker-attribution accuracy on eval/meetings/
uv run python scripts/learning_demo.py    # resolver before vs after 3 simulated corrections
```

## Layout

```
app/
├── stt.py          audio -> list[Utterance]; sha256-keyed .cache/stt/<hash>.json
├── speakers.py     diarized labels -> roster names (ECAPA + transcript fallback)
├── extract.py      ONE LLM call, tool-use/response_schema JSON, LLM_PROVIDER dispatch
├── feedback.py     SQLite: glossary + corrections, sklearn TF-IDF retrieval, running-mean voice embeddings
├── bench.py        full-pipeline eval -> bench/results.md
├── capture.py      AudioSource: FileSource, MicSource (sounddevice), SystemAudioSource stub
├── api/
│   ├── main.py     FastAPI: /, /process, /tts, /correct_task, /correct_speaker, /glossary
│   └── static/     index.html (vanilla JS SPA)
└── config.py       .env loader, typed accessors

eval/     audio + ground truth for bench
tests/    97 tests (STT grouping, Roman normalization, speaker resolve, feedback round-trip, extract rules, API, capture, smoke, bench)
scripts/  stand-alone evals and demos
```

See [AGENTS.md](AGENTS.md) for the hard constraints the project is built under
(no agent frameworks, no vector DB, Windows/Linux parity, every dependency
justified in `pyproject.toml`). [PROGRESS.md](PROGRESS.md) is the running log.
