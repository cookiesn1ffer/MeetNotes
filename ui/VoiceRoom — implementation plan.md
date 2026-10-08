# VoiceRoom — implementation plan

## Product shape

Build a single-screen meeting intelligence workspace for a local FastAPI backend. The experience is optimized for a three-minute projector demo: the main surface is always visible, the interface has a ready-to-show sample result, and live backend failures are surfaced without blanking the existing result.

## Implementation approach

- Use a dependency-light Vite app with semantic HTML, a small stateful JavaScript controller, and one CSS system.
- Keep the API boundary isolated in `src/api.js`. The adapter targets `POST /process`, `POST /correct_task`, `POST /correct_speaker`, and `POST /glossary`, with payload/response normalization helpers designed to be finalized when the FastAPI OpenAPI shapes are supplied.
- Use demo fixture data by default so the interface is usable before the local backend contract is pasted. A live FastAPI toggle targets `http://localhost:8000` (or `window.VOICEROOM_API_BASE` / local storage) and falls back with a clear error state.
- A staged runner animates the three processing phases and exposes timing values. Re-running the current recording keeps the previous result for comparison and raises the confidence/learning delta after corrections.
- Inline corrections update the local state immediately, send an adapter request, increment the learned counter on success, and show a compact confirmation. API errors are non-blocking and preserve the edited value for demo continuity.
- Browser speech is used as a reliable local fallback for the ElevenLabs-ready `Read my tasks` control. The adapter includes a `/tts` handoff point for a future FastAPI/ElevenLabs response without requiring a client-side API key.

## Project structure

- `index.html` — app shell, SEO metadata, route manifest link, and root mount.
- `src/main.js` — UI state, event wiring, rendering, staged processing, recorder, and interaction behavior.
- `src/api.js` — FastAPI base URL, request helpers, response normalization, demo/live mode adapter, and TTS handoff.
- `src/styles.css` — dark terminal-adjacent visual system, responsive layout, focus states, transitions, and projector contrast.
- `public/manus-routes.json` — route declaration for the single `/` page.
- `app.config.ts` — project logo metadata for the Webdev checkpoint.
- `package.json` — pinned Vite development/build scripts.

## Design system

### Design movement

**Terminal editorialism**: a quiet command-center interface influenced by high-end observability consoles, audio engineering bays, and monospace developer tools. It avoids SaaS cards, gradients, hero marketing, and ornamental illustration.

### Core principles

1. **Signal over chrome** — one accent, restrained borders, high information density, and no decorative dashboard fluff.
2. **Readable at distance** — oversized stage title, strong contrast, generous line height, and clear hierarchy for projector use.
3. **Editable truth** — every AI-derived claim is visibly mutable, with focused inputs and immediate learned feedback.
4. **Progress is a product feature** — processing stages and timings make speed and system work legible.

### Color philosophy

Near-black navy (`#0b0f14`) is the quiet room; slate surfaces (`#121922`, `#17212c`) separate work areas without boxed-in SaaS panels. A single mint-lime accent (`#b8f36b`) signals system health, active progress, and learned improvement. Amber and coral are reserved for deadlines and recoverable failures. Text is cool white with muted blue-grey secondary tones.

### Layout paradigm

A three-rail command surface: a fixed input rail, a flexible results canvas, and a narrow signal rail for learning/timings. On smaller screens the rails collapse into a readable vertical command sequence. Results stay above the fold with participant cards and transcript in one scrollable canvas.

### Signature elements

- The `V//ROOM` split-mark wordmark with a compact waveform/chevron glyph.
- Lime “signal rails” on active cards and stage rows, plus a thin progress beam.
- Small uppercase mono labels with `//` prefixes, like an instrument readout rather than a marketing UI.

### Interaction philosophy

Interactions should feel like editing a living transcript: click to focus, changes commit on blur/Enter, and confirmations appear at the edge of the changed object. No blocking modals for routine work. The primary Process action is explicit and calm; the Rerun action is always available after a correction.

### Animation

Use 150–260ms opacity/translate transitions for panels and controls. Processing uses a slow scanning beam and staged progress fill, never an infinite spinner. Record state uses a subtle pulse ring. Respect `prefers-reduced-motion` by disabling scan/pulse and preserving state changes.

### Typography system

Use a system sans stack for large UI copy (`Inter`, `ui-sans-serif`, `system-ui`) and a system monospace stack for labels, numbers, transcript metadata, and controls (`JetBrains Mono`, `SFMono-Regular`, `ui-monospace`). Headlines are tight and medium-weight; labels are uppercase with 0.14em tracking; transcript text is 16–18px for projector readability.

### Brand essence

**VoiceRoom turns bilingual meeting audio into accountable work, with corrections that compound.** Personality: **precise, calm, teachable**.

### Brand voice

Headlines are direct and technical: “Turn room noise into signal.” Microcopy shows the system at work: “Speaker map is settling.” and “Correction learned. Next pass will use it.” Avoid “Welcome”, “Get started”, or generic AI claims.

### Wordmark & logo

The logotype is `V//ROOM` in uppercase mono with a two-line waveform mark: two lime bars split by a dark slash, suggesting both a voice waveform and the Hindi/English handoff. The logo is rendered in CSS/SVG so it has no image dependency.

### Signature brand color

`#b8f36b` — an ownable signal-lime used sparingly for active state, progress, and learned improvement.

## Backend and fallback behavior

- `POST /process` receives the selected audio plus team/roster metadata through the adapter. The adapter currently sends a multipart request with `file`, `team_name`, and `roster` JSON fields, and can be adjusted in one place when the actual OpenAPI schema is pasted.
- Corrections send JSON with stable IDs and the edited values; normalized response data is merged when present.
- Glossary sends `{term, correction}` JSON.
- If the API is unreachable or returns a non-2xx response, retain the demo/result state, show an amber inline recovery banner, and keep the user’s correction in place.
- No file chosen, missing team name, and missing roster are validated in the input rail with explicit messages.

## Serving

The Vite server listens on the Webdev runtime port 3000 and binds to `0.0.0.0` for Preview. The app is static and has no managed server/database feature enabled. `npm run dev -- --host 0.0.0.0` is the development command; `npm run build` creates `dist/` for static publication if requested later.
