# 3-minute MeetNotes demo

Target: a stakeholder-safe walkthrough that fits a single screen share without
cuts. Five beats, times are cumulative.

## Setup (do this 2 min before the camera is on)

- `.env` has `ELEVENLABS_API_KEY`, `GEMINI_API_KEY`, and (optional)
  `ELEVENLABS_VOICE_ID`.
- Pre-seed one glossary term so Scribe doesn't fight the demo:
  ```sh
  curl -X POST http://127.0.0.1:8000/glossary \
    -H 'content-type: application/json' \
    -d '{"team_id":"demo","term":"Kaushal","kind":"person"}'
  ```
- Delete any stale correction history for the demo team so step 4 lands crisply:
  ```sh
  sqlite3 data/meetnotes.db "DELETE FROM corrections WHERE team_id='demo';
                             DELETE FROM speaker_embeddings WHERE team_id='demo';"
  ```
- Also clear the STT cache for the demo clip (optional — hitting cache makes
  the re-run instant, which is actually a nicer story if you keep it):
  ```sh
  rm -rf .cache/stt
  ```
- Start the server:
  ```sh
  uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
  ```
- Open `http://127.0.0.1:8000` in Firefox or Chromium.
- Have `eval/audio/demo-planets.wav` ready — a ~60 s Hinglish clip where
  **Aarush** says, in English with Hindi inflection:
  > "Sun ek dwarf star hai. Koshal, you are assigned to make a ppt on planets
  > for next meeting. Priya, launch readiness checklist kal tak chahiye."
  The clip intentionally mispronounces **Kaushal** as **Koshal** — that's the
  mistake we'll fix on camera.

## Beat 1 — play the Hinglish clip (0:00 – 0:25)

- In the UI, type `demo` into **Team ID** and `Aarush, Kaushal, Priya` into
  **Roster**.
- Hit the system player once on `demo-planets.wav` so the audience hears the
  code-switching. Pause.
- Say: "This clip is one speaker, English with Hindi inflection, a planted
  name mispronunciation, and two assignments. Standard meeting."
- Pick the file in the **Audio file** input. Click **Process**.

## Beat 2 — show the per-person task list (0:25 – 1:05)

- The results panel appears. Call out, in order:
  - **Summary** — the LLM's two-sentence recap. "Preserves Hinglish."
  - **Tasks by person** — grouped by assignee.
    - Under *Koshal*: `ppt on planets (next meeting)` — flag this as the typo
      we're about to fix.
    - Under *Priya*: `launch readiness checklist (कल तक)` — same clip, both
      people, correct deadline preserved as spoken.
  - **Transcript** — point out that Devanagari (कल तक) is retained as spoken.
  - **Timings** — scroll to the `Timings` panel: `stage=stt`, `stage=speakers`,
    `stage=extract`, each with elapsed seconds. Say: "Three stages, one LLM call
    per meeting, no chains."

## Beat 3 — correct one task and one speaker (1:05 – 2:00)

- Under *Koshal*, click **edit** on the ppt task.
  - Change **Assignee** from `Koshal` to `Kaushal`.
  - Leave the task and deadline alone.
  - Click **save**. The status line reads *"Correction saved. Re-process to apply."*
  - Narrate: "That POST hit `/correct_task`. It's now in SQLite, scoped to this
    team. The fuzzy matcher already handles Koshal/Kaushal, but we're going to
    teach the system the specific mapping so it carries across clips."
- Fire a speaker correction from a terminal (voice embedding — the UI edits
  task text; this one changes who the voice belongs to):
  ```sh
  curl -X POST http://127.0.0.1:8000/correct_speaker \
    -H 'content-type: application/json' \
    -d '{"team_id":"demo","before":"speaker_0","after":"Aarush",
         "audio_path":"'$PWD'/eval/audio/demo-planets.wav","start":0.0,"end":4.0}'
  ```
  - Narrate: "That just ran ECAPA on the first four seconds of the clip, folded
    the vector into a running mean under `(demo, Aarush)`, and persisted it.
    Next meeting with Aarush in it, the speaker resolver finds him by voice
    before it ever reads the transcript."

## Beat 4 — rerun the same clip, show the fix persists (2:00 – 2:30)

- Click **Process** again on the same `demo-planets.wav`.
  - Point at **Timings**: `stage=stt elapsed_s=0.0… cached=True`. "Same audio
    file, same sha256, cache hit — zero ElevenLabs calls on this re-run."
  - The **Tasks by person** list now puts the ppt task under **Kaushal**, not
    Koshal. The transcript's `Koshal` is still raw (that's what Scribe heard),
    but the extraction mapped it correctly because the correction landed in
    the prompt as a `wrong -> corrected` few-shot example (TF-IDF-retrieved
    on the current transcript).
  - "That's the loop. Correct it once, it sticks for the team."

## Beat 5 — benchmark + TTS readout (2:30 – 3:00)

- Switch to a terminal:
  ```sh
  uv run python -m app.bench
  cat bench/results.md
  ```
  - Columns: `time-to-notes`, `STT time`, `LLM time`, `WER`, `speaker acc`,
    `task precision`, `task recall`. One row per clip in `eval/audio/`.
  - Call the thresholds from the footer line verbatim: "time-to-notes ≤ 0.25 ×
    duration (so under 15 seconds for a 60-second clip), speaker accuracy
    at least 90 %, task recall at least 80 %."
  - If this is your first bench run, say: "Numbers populate as we add clips to
    `eval/audio/` and `eval/truth/`."
- Back in the UI, under **Kaushal**, click **🔊 read my tasks**.
  - ElevenLabs TTS plays through the browser. Narrate: "That request hit
    `/tts`, which proxies ElevenLabs. Same key you already have for Scribe.
    This is Kaushal's followups spoken back in natural voice — useful at the
    end of a meeting when nobody wants to re-watch the recording."
- Close with: "One meeting in, team-scoped glossary + per-person voice
  signatures out, every correction improves the next run. The whole pipeline
  is four modules, no agent framework, no vector DB."

## Fallbacks (keep handy)

- **If the Scribe call stalls**: hit **Process** again — the cache will serve
  the previous response if one exists for the same clip. Otherwise narrate
  Beat 2 from a dry run you did before camera.
- **If TTS fails** (bad voice id, over-quota): skip Beat 5's audio, show
  `bench/results.md` on-screen, and point at the two tested correction paths
  (`/correct_task` and `/correct_speaker`) as the "it learns" story.
- **If the LLM doesn't apply the correction** on the re-run (Gemini can be
  stubborn), switch providers with `LLM_PROVIDER=anthropic uv run uvicorn ...`
  and rerun — Claude follows few-shot examples very reliably.
