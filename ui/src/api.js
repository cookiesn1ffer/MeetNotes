// Data layer between the UI and the FastAPI backend (app/api/main.py).
// Request/response shapes here mirror the backend exactly; the backend is the only data source.

const DEV_BACKEND = 'http://localhost:8000';

export function getApiBase() {
  const saved = window.VOICEROOM_API_BASE || localStorage.getItem('voiceroom_api_base');
  if (saved) return saved.trim().replace(/\/$/, '');
  // Served by the backend itself (`python -m app serve`): same origin. Otherwise (`pnpm dev`
  // on another port) talk to the backend on :8000, which allows localhost origins via CORS.
  return location.port === '8000' || location.port === '' ? '' : DEV_BACKEND;
}

export function setApiBase(url) {
  const value = (url || '').trim().replace(/\/$/, '');
  if (value) localStorage.setItem('voiceroom_api_base', value);
  else localStorage.removeItem('voiceroom_api_base');
}

// FastAPI errors are {"detail": "<message>"} (or a list of validation errors).
async function errorMessage(response) {
  const text = await response.text().catch(() => '');
  try {
    const { detail } = JSON.parse(text);
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((item) => item.msg).join('; ');
  } catch (_) { /* not JSON */ }
  return text.slice(0, 160) || `${response.status} ${response.statusText}`;
}

async function send(path, options) {
  let response;
  try {
    response = await fetch(`${getApiBase()}${path}`, options);
  } catch (_) {
    throw new Error(`Cannot reach the backend at ${getApiBase() || location.origin}. Is \`python -m app serve\` running?`);
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

async function postJson(path, payload) {
  const response = await send(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function processMeeting({ file, teamName, roster }) {
  const body = new FormData();
  body.append('audio', file, file.name || 'meeting-audio');
  body.append('team_id', teamName);
  body.append('roster', roster.join(','));
  const response = await send('/process', { method: 'POST', body });
  return normalizeProcessResponse(await response.json());
}

// UI task {text, assigner, assignee, deadline} <-> backend task {task, assigned_by, assignee, deadline}.
const toBackendTask = (task) => ({
  task: task.text,
  assigned_by: task.assigner,
  assignee: task.assignee,
  deadline: task.deadline === '—' ? null : task.deadline
});

export function correctTask({ teamName, before, after }) {
  return postJson('/correct_task', {
    team_id: teamName,
    before: toBackendTask(before),
    after: toBackendTask(after),
    source_quote: after.sourceQuote || before.sourceQuote || ''
  });
}

// Resolves to {id, voice_learned}: voice_learned is false when voice embeddings are unavailable.
export function correctSpeaker({ teamName, runId, line, from, to }) {
  return postJson('/correct_speaker', {
    team_id: teamName,
    before: from,
    after: to,
    run_id: runId,
    start: line.start,
    end: line.end
  });
}

export function addGlossary({ teamName, term }) {
  return postJson('/glossary', { team_id: teamName, term, kind: 'term' });
}

// The backend proxies ElevenLabs (the API key never reaches the browser) and returns audio bytes.
export async function synthesizeSpeech(text) {
  const response = await send('/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return URL.createObjectURL(await response.blob());
}

function normalizeProcessResponse(payload) {
  if (!Array.isArray(payload?.people) || !Array.isArray(payload?.transcript)) {
    throw new Error('Unexpected response from /process (missing people or transcript).');
  }
  return {
    runId: payload.run_id,
    confidence: payload.confidence ?? null,
    timings: payload.stage_timings,
    people: payload.people.map(normalizePerson),
    transcript: payload.transcript.map(normalizeTranscriptLine)
  };
}

function normalizePerson(person, index) {
  return {
    id: person.id,
    name: person.name,
    initials: person.name.split(' ').map((part) => part[0]).join('').slice(0, 2).toUpperCase(),
    color: ['lime', 'violet', 'amber', 'cyan'][index % 4],
    notes: person.notes,
    tasks: person.tasks.map(normalizeTask)
  };
}

function normalizeTask(task) {
  return {
    id: task.id,
    text: task.task,
    assigner: task.assigned_by,
    assignee: task.assignee,
    deadline: task.deadline || '—',
    sourceQuote: task.source_quote
  };
}

function normalizeTranscriptLine(line) {
  return {
    id: line.id,
    speaker: line.speaker,
    text: line.text,
    language: line.language,
    timestamp: line.timestamp,
    start: line.start,
    end: line.end
  };
}
