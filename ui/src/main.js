import * as api from './api.js';
import './styles.css';

const emptyResult = () => ({
  runId: '—',
  confidence: null,
  timings: { transcribe: 0, speakers: 0, tasks: 0, total: 0 },
  people: [],
  transcript: []
});

const state = {
  result: emptyResult(),
  previousResult: null,
  roster: [],
  teamName: '',
  file: null,
  fileName: '',
  processing: false,
  activeStage: 3,
  hasRerun: false,
  compareOpen: false,
  transcriptOpen: true,
  apiBase: api.getApiBase(),
  learned: 0,
  glossary: [],
  error: '',
  toast: '',
  recording: false,
  mediaRecorder: null,
  chunks: [],
  validation: ''
};

const stages = [
  { key: 'transcribe', label: 'Transcribe audio', detail: 'Finding words + language switches' },
  { key: 'speakers', label: 'Map speakers', detail: 'Separating voices into a speaker graph' },
  { key: 'tasks', label: 'Extract tasks', detail: 'Turning decisions into accountable work' }
];

const app = document.querySelector('#app');

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

const pct = (value) => (value === null || value === undefined ? '—' : `${value}%`);

function languageBadge() {
  const tags = state.result.transcript.map((line) => line.language);
  const hindi = tags.some((tag) => tag.includes('HI'));
  const english = tags.some((tag) => tag.includes('EN'));
  if (hindi && english) return 'Hindi + English detected';
  if (hindi) return 'Hindi detected';
  return english ? 'English detected' : '';
}

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

function render() {
  const timings = state.result.timings;
  const total = timings.total || 0;
  const hasAudio = Boolean(state.file);
  const beforeConfidence = state.previousResult?.confidence ?? null;
  const afterConfidence = state.result.confidence;
  const confidenceDelta = beforeConfidence !== null && afterConfidence !== null ? afterConfidence - beforeConfidence : null;
  const timingWidth = (value) => (total > 0 ? Math.round((value / total) * 100) : 0);
  const completed = state.processing ? Math.min(96, ((state.activeStage + 0.45) / stages.length) * 100) : 100;
  app.innerHTML = `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand-lockup">
          <div class="brand-mark" aria-hidden="true"><i></i><i></i><b></b></div>
          <div><div class="wordmark">V<span>//</span>ROOM</div><div class="brand-sub">MEETING INTELLIGENCE / 01</div></div>
        </div>
        <div class="topbar-status">
          <span class="status-dot ${state.processing ? 'is-processing' : ''}"></span>
          <span>${state.processing ? 'PIPELINE ACTIVE' : 'LOCAL PIPELINE READY'}</span>
          <span class="slash">//</span>
          <span class="muted">${escapeHtml(state.apiBase || location.origin)}</span>
        </div>
        <div class="topbar-right">
          <span class="learned-pill"><span class="mini-spark">✦</span> ${state.learned} learned</span>
          <span class="latency">${total.toFixed(1)}s total</span>
        </div>
      </header>

      <div class="layout-grid">
        <aside class="input-rail">
          <div class="rail-heading"><span class="eyebrow">01 / INPUT</span><span class="rail-state">${hasAudio ? 'READY' : 'WAITING'}</span></div>
          <h1>Turn room noise<br /><em>into signal.</em></h1>
          <p class="lede">Drop a recording, name the room, and let the pipeline find the work hiding inside.</p>

          <section class="input-section">
            <div class="section-label"><span>Audio source</span><span class="required">required</span></div>
            <div class="dropzone ${hasAudio ? 'has-file' : ''} ${state.recording ? 'is-recording' : ''}" data-dropzone tabindex="0">
              <input id="file-input" type="file" accept="audio/*" hidden />
              <div class="drop-icon" aria-hidden="true">${state.recording ? '<span class="record-ring"></span>' : '↥'}</div>
              <div class="drop-copy">
                <strong>${state.recording ? 'Recording from mic…' : hasAudio ? escapeHtml(state.fileName) : 'Drop meeting audio here'}</strong>
                <small>${state.recording ? 'Click stop when you are done' : hasAudio ? 'Ready for a fresh pass' : 'MP3, WAV, M4A · up to 250MB'}</small>
              </div>
              <button class="icon-button" data-action="browse" aria-label="Browse for audio">↗</button>
            </div>
            <div class="audio-actions">
              <button class="subtle-button" data-action="record">${state.recording ? '■ Stop recording' : '◉ Record from mic'}</button>
              ${hasAudio ? `<button class="text-button" data-action="clear-file">clear</button>` : ''}
            </div>
          </section>

          <section class="input-section compact-section">
            <label class="section-label" for="team-name"><span>Team / room name</span><span class="required">required</span></label>
            <input class="field" id="team-name" value="${escapeHtml(state.teamName)}" placeholder="e.g. Product / India" />
          </section>

          <section class="input-section compact-section">
            <div class="section-label"><span>Who is in the room?</span><span class="required">required</span></div>
            <div class="roster-box">
              <div class="chip-row">${state.roster.map((person, index) => `<span class="person-chip"><span class="chip-avatar avatar-${index % 4}">${escapeHtml(person.split(' ').map((p) => p[0]).join('').slice(0, 2))}</span>${escapeHtml(person)}<button data-action="remove-roster" data-index="${index}" aria-label="Remove ${escapeHtml(person)}">×</button></span>`).join('')}</div>
              <input id="roster-input" class="chip-input" placeholder="Add a name + Enter" />
            </div>
          </section>

          <div class="validation-message ${state.validation ? 'show' : ''}">${state.validation || 'Audio is kept locally so voice corrections can learn from it.'}</div>

          <div class="rail-bottom">
            <button class="primary-button process-button" data-action="process" ${state.processing ? 'disabled' : ''}>
              <span>${state.processing ? 'Processing…' : state.hasRerun ? 'Re-run same recording' : 'Process meeting'}</span><b>↗</b>
            </button>
            <div class="mode-panel">
              <div class="mode-row"><span class="tiny-label">API</span><span class="switch-label">FastAPI</span></div>
              <input id="api-base" class="field field-small" value="${escapeHtml(state.apiBase)}" aria-label="FastAPI base URL" />
            </div>
          </div>
        </aside>

        <main class="results-canvas">
          <div class="canvas-topline">
            <div><span class="eyebrow">02 / OUTPUT</span><h2>${state.processing ? 'The room is becoming legible.' : 'Meeting signal, sorted.'}</h2></div>
            <div class="output-actions"><span class="run-label">RUN <strong>${escapeHtml(state.result.runId)}</strong></span><button class="ghost-button ${state.transcriptOpen ? 'active' : ''}" data-action="toggle-transcript">${state.transcriptOpen ? 'Hide' : 'Show'} transcript</button></div>
          </div>
          ${state.error ? `<div class="error-banner"><span class="error-icon">!</span><div><strong>${escapeHtml(state.error.title || 'Backend unavailable')}</strong><span>${escapeHtml(state.error.message || state.error)}</span></div><button data-action="dismiss-error">×</button></div>` : ''}
          <div class="progress-strip ${state.processing ? 'processing' : ''}"><span style="width:${completed}%"></span></div>
          ${state.processing ? processingView() : resultsView()}
        </main>

        <aside class="signal-rail">
          <div class="rail-heading"><span class="eyebrow">03 / SIGNAL</span><span class="rail-state">LIVE</span></div>
          <section class="signal-card confidence-card">
            <div class="card-kicker"><span class="accent-dot"></span> SPEAKER MAP CONFIDENCE</div>
            <div class="confidence-value">${afterConfidence ?? '—'}<small>${afterConfidence === null ? '' : '%'}</small></div>
            <div class="confidence-bar"><span style="width:${afterConfidence ?? 0}%"></span></div>
            <div class="confidence-foot"><span>${afterConfidence === null ? 'no run yet' : afterConfidence >= 80 ? 'high signal' : 'needs review'}</span><span>${state.hasRerun && confidenceDelta !== null ? `${confidenceDelta >= 0 ? '+' : ''}${confidenceDelta.toFixed(1)}% since correction` : 'baseline run'}</span></div>
          </section>
          <section class="signal-card timings-card">
            <div class="card-kicker"><span class="accent-dot dim"></span> TIMINGS / ${total.toFixed(1)}s TOTAL</div>
            <div class="timing-list">
              ${timingRow('01', 'Transcribe', timings.transcribe, timingWidth(timings.transcribe), 'lime')}
              ${timingRow('02', 'Map speakers', timings.speakers, timingWidth(timings.speakers), 'violet')}
              ${timingRow('03', 'Extract tasks', timings.tasks, timingWidth(timings.tasks), 'amber')}
            </div>
          </section>
          <section class="signal-card learning-card">
            <div class="card-kicker"><span class="accent-dot"></span> CORRECTION MEMORY</div>
            <div class="learning-head"><strong>${state.learned === 0 ? 'Ready to learn.' : `${state.learned} correction${state.learned === 1 ? '' : 's'} learned.`}</strong><span>✦</span></div>
            <p>${state.learned === 0 ? 'Edit a task, speaker, or note. VoiceRoom will carry that signal into the next pass.' : 'The next pass will use these edits to tighten the speaker map and task ownership.'}</p>
            <div class="learning-compare"><span>before</span><b>${pct(beforeConfidence)}</b><i></i><span>after</span><b class="after-value">${state.hasRerun ? pct(afterConfidence) : '—'}</b></div>
            ${state.hasRerun ? '<button class="text-button compare-button" data-action="toggle-compare">view before / after →</button>' : ''}
          </section>
          <section class="signal-card glossary-card">
            <div class="card-kicker"><span class="accent-dot dim"></span> GLOSSARY</div>
            <p class="small-copy">Names and terms to keep intact.</p>
            <div class="glossary-list">${state.glossary.map((term) => `<span>${escapeHtml(term)}</span>`).join('')}</div>
            <form id="glossary-form" class="glossary-form"><input name="term" placeholder="Add term" aria-label="Add glossary term" required /><button aria-label="Add glossary term">+</button></form>
          </section>
        </aside>
      </div>
      ${state.toast ? `<div class="toast" role="status"><span>✦</span>${escapeHtml(state.toast)}</div>` : ''}
      <footer class="footer-line"><span>VOICE + LANGUAGE + ACCOUNTABILITY</span><span>built for the room, not the feed</span><span>v0.4 / ${new Date().getFullYear()}</span></footer>
    </div>
  `;
}

function timingRow(index, label, value, width, color) {
  return `<div class="timing-row"><div class="timing-meta"><span>${index}</span><strong>${label}</strong><b>${value.toFixed(1)}s</b></div><div class="timing-bar"><i class="bar-${color}" style="width:${width}%"></i></div></div>`;
}

function processingView() {
  return `<section class="processing-view"><div class="process-orbit"><span></span><span></span><span></span><b>V//R</b></div><div class="processing-copy"><span class="eyebrow">PIPELINE / ${String(state.activeStage + 1).padStart(2, '0')} OF 03</span><h3>${escapeHtml(stages[Math.min(state.activeStage, 2)].label)}</h3><p>${escapeHtml(stages[Math.min(state.activeStage, 2)].detail)}<span class="ellipsis">...</span></p></div><div class="stage-list">${stages.map((stage, index) => `<div class="stage-row ${index < state.activeStage ? 'done' : ''} ${index === state.activeStage ? 'active' : ''}"><span class="stage-number">${index < state.activeStage ? '✓' : `0${index + 1}`}</span><div><strong>${stage.label}</strong><small>${index < state.activeStage ? 'complete' : index === state.activeStage ? 'working now' : 'queued'}</small></div><span class="stage-mark">${index < state.activeStage ? '●' : index === state.activeStage ? '◌' : '○'}</span></div>`).join('')}</div></section>`;
}

function resultsView() {
  return `<section class="results-stack">
    <div class="result-meta-row"><span><b>${state.result.people.length}</b> people / <b>${state.result.people.reduce((sum, person) => sum + person.tasks.length, 0)}</b> tasks found</span>${languageBadge() ? `<span class="language-badge"><i></i> ${languageBadge()}</span>` : ''}</div>
    ${state.result.people.length ? '' : '<p class="lede">Process a meeting to see per-person tasks and notes here.</p>'}
    ${state.compareOpen ? comparisonView() : ''}
    <div class="people-grid">${state.result.people.map(personCard).join('')}</div>
    ${state.transcriptOpen ? transcriptView() : ''}
  </section>`;
}

function personCard(person) {
  const taskCount = person.tasks.length;
  return `<article class="person-card accent-${person.color}">
    <div class="person-head"><div class="person-id"><span class="person-avatar">${escapeHtml(person.initials)}</span><div><h3>${escapeHtml(person.name)}</h3><span>${taskCount} task${taskCount === 1 ? '' : 's'} / owner view</span></div></div><button class="voice-button" data-action="read-tasks" data-person-id="${escapeHtml(person.id)}"><span>◖</span> Read my tasks</button></div>
    <div class="notes-block"><span class="mini-label">NOTES <em>editable</em></span><textarea data-note-person="${escapeHtml(person.id)}" aria-label="Notes for ${escapeHtml(person.name)}">${escapeHtml(person.notes)}</textarea></div>
    <div class="tasks-block"><div class="task-header"><span class="mini-label">TASKS / ${String(taskCount).padStart(2, '0')}</span><span class="task-hint">click any field to edit</span></div>
      ${person.tasks.map((task) => taskRow(person, task)).join('')}
    </div>
  </article>`;
}

function taskRow(person, task) {
  const options = unique([...state.roster, 'Meeting']).map((name) => `<option ${name === task.assignee ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('');
  return `<div class="task-row" data-task-row="${escapeHtml(task.id)}"><span class="task-check">□</span><div class="task-fields"><input class="task-text" data-task-field="text" data-person-id="${escapeHtml(person.id)}" data-task-id="${escapeHtml(task.id)}" value="${escapeHtml(task.text)}" aria-label="Task text" /><div class="task-detail-fields"><label>by <input data-task-field="assigner" data-person-id="${escapeHtml(person.id)}" data-task-id="${escapeHtml(task.id)}" value="${escapeHtml(task.assigner)}" aria-label="Task assigner" /></label><label>to <select data-task-field="assignee" data-person-id="${escapeHtml(person.id)}" data-task-id="${escapeHtml(task.id)}" aria-label="Task assignee">${options}</select></label><label>due <input data-task-field="deadline" data-person-id="${escapeHtml(person.id)}" data-task-id="${escapeHtml(task.id)}" value="${escapeHtml(task.deadline)}" aria-label="Task deadline" /></label></div></div></div>`;
}

function transcriptView() {
  return `<section class="transcript-panel"><div class="panel-heading"><div><span class="eyebrow">TRANSCRIPT / CODE-SWITCH TRACE</span><h3>What the room actually said</h3></div><span class="transcript-note"><i></i> language mix visible</span></div><div class="transcript-list">${state.result.transcript.map((line) => `<div class="transcript-line"><time>${escapeHtml(line.timestamp)}</time><select data-transcript-speaker="${escapeHtml(line.id)}" aria-label="Speaker label for ${escapeHtml(line.timestamp)}">${unique(state.roster).map((name) => `<option ${name === line.speaker ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('')}<option ${!state.roster.includes(line.speaker) ? 'selected' : ''}>${escapeHtml(line.speaker)}</option></select><span class="language-tag ${line.language.includes('HI') ? 'mixed' : ''}">${escapeHtml(line.language)}</span><p>${highlightHindi(line.text)}</p></div>`).join('')}</div></section>`;
}

function highlightHindi(text) {
  return escapeHtml(text).replace(/(Toh|Haan|Theek hai)/g, '<mark>$1</mark>');
}

function comparisonView() {
  const before = state.previousResult?.confidence ?? null;
  const after = state.result.confidence;
  return `<div class="comparison-panel"><div><span class="eyebrow">BEFORE / AFTER</span><strong>Correction loop closed.</strong><p>Speaker map and ownership improved after ${state.learned} learned correction${state.learned === 1 ? '' : 's'}.</p></div><div class="compare-score"><span>${pct(before)}</span><b>→</b><strong>${pct(after)}</strong></div><button data-action="toggle-compare" class="text-button">close ×</button></div>`;
}

function unique(items) { return [...new Set(items.filter(Boolean))]; }

function validate() {
  state.teamName = document.querySelector('#team-name')?.value.trim() || '';
  if (!state.file) return 'Add an audio file or record the meeting first.';
  if (!state.teamName) return 'Name the team or room so the summary has context.';
  if (state.roster.length < 1) return 'Add at least one person to the roster.';
  return '';
}

async function runProcessing() {
  state.validation = validate();
  if (state.validation) { render(); return; }
  const rerun = state.hasRerun || state.learned > 0;
  state.processing = true;
  state.error = '';
  state.activeStage = 0;
  render();

  // The stage list is only a progress indicator while the single /process request runs;
  // the timings shown afterwards are the backend's own per-stage measurements.
  const request = api.processMeeting({ file: state.file, teamName: state.teamName, roster: state.roster });
  const settled = request.then(() => true, () => true);
  let finished = false;
  settled.then(() => { finished = true; });

  try {
    for (let index = 0; index < stages.length && !finished; index += 1) {
      state.activeStage = index;
      render();
      await Promise.race([sleep(1500), settled]);
    }
    const nextResult = await request;
    state.previousResult = state.result.people.length ? state.result : null;
    state.result = nextResult;
    state.hasRerun = rerun;
    state.activeStage = 3;
    showToast(rerun ? 'Fresh pass complete. The correction is visible.' : 'Meeting processed. Review the signal below.');
  } catch (error) {
    const kept = state.result.people.length ? ' Your previous result is still shown.' : '';
    state.error = { title: 'Processing failed', message: `${error.message}${kept}` };
    state.activeStage = 3;
  } finally {
    state.processing = false;
    render();
  }
}

function showToast(message) {
  state.toast = message;
  render();
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { state.toast = ''; render(); }, 3200);
}

async function commitTask(target) {
  const person = state.result.people.find((item) => item.id === target.dataset.personId);
  const task = person?.tasks.find((item) => item.id === target.dataset.taskId);
  if (!task) return;
  const nextValue = target.value.trim();
  if (!nextValue || task[target.dataset.taskField] === nextValue) return;
  const before = { ...task };
  task[target.dataset.taskField] = nextValue;
  try {
    await api.correctTask({ teamName: state.teamName, before, after: { ...task } });
    state.learned += 1;
    showToast('Correction learned. Next pass will use it.');
  } catch (error) {
    state.error = { title: 'Correction not saved', message: `${error.message}. The edit is only on this page.` };
    render();
  }
}

// Notes have no backend correction path (they are not used for learning), so edits stay on the page.
function commitNote(target) {
  const person = state.result.people.find((item) => item.id === target.dataset.notePerson);
  if (!person || person.notes === target.value) return;
  person.notes = target.value;
  showToast('Note updated on this page. Notes are not used for learning.');
}

async function commitSpeaker(target) {
  const line = state.result.transcript.find((item) => item.id === target.dataset.transcriptSpeaker);
  if (!line || line.speaker === target.value) return;
  const from = line.speaker;
  line.speaker = target.value;
  try {
    const saved = await api.correctSpeaker({ teamName: state.teamName, runId: state.result.runId, line, from, to: line.speaker });
    state.learned += 1;
    showToast(saved.voice_learned ? 'Speaker label learned. Voice updated.' : 'Speaker label saved. Voice not updated (voice embeddings unavailable).');
  } catch (error) {
    state.error = { title: 'Speaker fix not saved', message: `${error.message}. The label is only changed on this page.` };
    render();
  }
}

function chooseFile(file) {
  if (!file) return;
  state.file = file;
  state.fileName = file.name;
  state.validation = '';
  render();
  showToast(`${file.name} is ready for processing.`);
}

async function toggleRecording() {
  if (state.recording && state.mediaRecorder) {
    state.mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    state.error = { title: 'Microphone recording is not available', message: 'Drop an audio file instead, or open this preview in a browser with microphone permissions.' };
    render();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    state.chunks = [];
    state.mediaRecorder = recorder;
    recorder.ondataavailable = (event) => state.chunks.push(event.data);
    recorder.onstop = () => {
      const blob = new Blob(state.chunks, { type: 'audio/webm' });
      state.file = new File([blob], 'voice-room-recording.webm', { type: 'audio/webm' });
      state.fileName = 'voice-room-recording.webm';
      state.recording = false;
      stream.getTracks().forEach((track) => track.stop());
      render();
      showToast('Recording captured. Ready for processing.');
    };
    recorder.start();
    state.recording = true;
    render();
  } catch (error) {
    state.error = { title: 'Microphone permission was not granted', message: 'You can still drop an audio file to continue.' };
    render();
  }
}

async function readTasks(personId) {
  const person = state.result.people.find((item) => item.id === personId);
  if (!person) return;
  const text = person.tasks.length
    ? `${person.name}'s tasks. ${person.tasks.map((task) => `${task.text}, due ${task.deadline}`).join('. ')}.`
    : `${person.name} has no tasks.`;
  try {
    const audioUrl = await api.synthesizeSpeech(text);
    const audio = new Audio(audioUrl);
    audio.onended = () => URL.revokeObjectURL(audioUrl);
    await audio.play();
    showToast('Playing ElevenLabs task readout.');
    return;
  } catch (error) {
    // Falling back to the browser voice keeps the readout usable without ElevenLabs.
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.94;
      window.speechSynthesis.speak(utterance);
      showToast(`ElevenLabs unavailable (${error.message}). Playing the browser voice.`);
    } else {
      state.error = { title: 'Voice playback failed', message: error.message };
      render();
    }
  }
}

async function submitGlossary(form) {
  const data = new FormData(form);
  const term = String(data.get('term') || '').trim();
  if (!term) return;
  const teamName = document.querySelector('#team-name')?.value.trim() || state.teamName;
  if (!teamName) { state.validation = 'Name the team or room first so the term is saved for it.'; render(); return; }
  try {
    await api.addGlossary({ teamName, term });
    state.glossary = unique([...state.glossary, term]);
    form.reset();
    showToast(`${term} added to the glossary.`);
  } catch (error) {
    state.error = { title: 'Glossary term not saved', message: error.message };
    render();
  }
}

document.addEventListener('click', async (event) => {
  const actionTarget = event.target.closest('[data-action]');
  if (!actionTarget) return;
  const action = actionTarget.dataset.action;
  if (action === 'browse') document.querySelector('#file-input')?.click();
  if (action === 'record') await toggleRecording();
  if (action === 'clear-file') { state.file = null; state.fileName = ''; render(); }
  if (action === 'process') await runProcessing();
  if (action === 'remove-roster') { state.roster.splice(Number(actionTarget.dataset.index), 1); render(); }
  if (action === 'toggle-transcript') { state.transcriptOpen = !state.transcriptOpen; render(); }
  if (action === 'toggle-compare') { state.compareOpen = !state.compareOpen; render(); }
  if (action === 'dismiss-error') { state.error = ''; render(); }
  if (action === 'read-tasks') await readTasks(actionTarget.dataset.personId);
});

document.addEventListener('change', async (event) => {
  const target = event.target;
  if (target.matches('#file-input')) chooseFile(target.files?.[0]);
  if (target.matches('[data-task-field]')) await commitTask(target);
  if (target.matches('[data-transcript-speaker]')) await commitSpeaker(target);
});

document.addEventListener('focusout', async (event) => {
  const target = event.target;
  if (target.matches('[data-task-field]')) await commitTask(target);
  if (target.matches('[data-note-person]')) commitNote(target);
  if (target.matches('#api-base')) { api.setApiBase(target.value); state.apiBase = api.getApiBase(); }
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && event.target.matches('#roster-input')) {
    event.preventDefault();
    const name = event.target.value.trim();
    if (name && !state.roster.includes(name)) { state.roster.push(name); event.target.value = ''; render(); document.querySelector('#roster-input')?.focus(); }
  }
});

document.addEventListener('submit', async (event) => {
  if (event.target.matches('#glossary-form')) { event.preventDefault(); await submitGlossary(event.target); }
});

document.addEventListener('dragover', (event) => {
  if (event.target.closest('[data-dropzone]')) { event.preventDefault(); event.target.closest('[data-dropzone]').classList.add('is-dragging'); }
});
document.addEventListener('dragleave', (event) => { event.target.closest('[data-dropzone]')?.classList.remove('is-dragging'); });
document.addEventListener('drop', (event) => {
  const zone = event.target.closest('[data-dropzone]');
  if (!zone) return;
  event.preventDefault();
  zone.classList.remove('is-dragging');
  chooseFile(event.dataTransfer.files?.[0]);
});

document.addEventListener('click', (event) => {
  if (event.target.closest('[data-dropzone]') && !event.target.closest('button')) document.querySelector('#file-input')?.click();
});

render();
