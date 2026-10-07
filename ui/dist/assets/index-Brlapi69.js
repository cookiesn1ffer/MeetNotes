(function(){const s=document.createElement("link").relList;if(s&&s.supports&&s.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))n(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const d of o.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&n(d)}).observe(document,{childList:!0,subtree:!0});function t(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function n(i){if(i.ep)return;i.ep=!0;const o=t(i);fetch(i.href,o)}})();const N="http://localhost:8000";function m(){const e=window.VOICEROOM_API_BASE||localStorage.getItem("voiceroom_api_base");return e?e.trim().replace(/\/$/,""):location.port==="8000"||location.port===""?"":N}function A(e){const s=(e||"").trim().replace(/\/$/,"");s?localStorage.setItem("voiceroom_api_base",s):localStorage.removeItem("voiceroom_api_base")}async function I(e){const s=await e.text().catch(()=>"");try{const{detail:t}=JSON.parse(s);if(typeof t=="string")return t;if(Array.isArray(t))return t.map(n=>n.msg).join("; ")}catch{}return s.slice(0,160)||`${e.status} ${e.statusText}`}async function f(e,s){let t;try{t=await fetch(`${m()}${e}`,s)}catch{throw new Error(`Cannot reach the backend at ${m()||location.origin}. Is \`python -m app serve\` running?`)}if(!t.ok)throw new Error(await I(t));return t}async function h(e,s){return(await f(e,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)})).json()}async function O({file:e,teamName:s,roster:t}){const n=new FormData;n.append("audio",e,e.name||"meeting-audio"),n.append("team_id",s),n.append("roster",t.join(","));const i=await f("/process",{method:"POST",body:n});return C(await i.json())}const w=e=>({task:e.text,assigned_by:e.assigner,assignee:e.assignee,deadline:e.deadline==="—"?null:e.deadline});function L({teamName:e,before:s,after:t}){return h("/correct_task",{team_id:e,before:w(s),after:w(t),source_quote:t.sourceQuote||s.sourceQuote||""})}function P({teamName:e,runId:s,line:t,from:n,to:i}){return h("/correct_speaker",{team_id:e,before:n,after:i,run_id:s,start:t.start,end:t.end})}function x({teamName:e,term:s}){return h("/glossary",{team_id:e,term:s,kind:"term"})}async function M(e){const s=await f("/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:e})});return URL.createObjectURL(await s.blob())}function C(e){if(!Array.isArray(e==null?void 0:e.people)||!Array.isArray(e==null?void 0:e.transcript))throw new Error("Unexpected response from /process (missing people or transcript).");return{runId:e.run_id,confidence:e.confidence??null,timings:e.stage_timings,people:e.people.map(_),transcript:e.transcript.map(F)}}function _(e,s){return{id:e.id,name:e.name,initials:e.name.split(" ").map(t=>t[0]).join("").slice(0,2).toUpperCase(),color:["lime","violet","amber","cyan"][s%4],notes:e.notes,tasks:e.tasks.map(q)}}function q(e){return{id:e.id,text:e.task,assigner:e.assigned_by,assignee:e.assignee,deadline:e.deadline||"—",sourceQuote:e.source_quote}}function F(e){return{id:e.id,speaker:e.speaker,text:e.text,language:e.language,timestamp:e.timestamp,start:e.start,end:e.end}}const j=()=>({runId:"—",confidence:null,timings:{transcribe:0,speakers:0,tasks:0,total:0},people:[],transcript:[]}),a={result:j(),previousResult:null,roster:[],teamName:"",file:null,fileName:"",processing:!1,activeStage:3,hasRerun:!1,compareOpen:!1,transcriptOpen:!0,apiBase:m(),learned:0,glossary:[],error:"",toast:"",recording:!1,mediaRecorder:null,chunks:[],validation:""},u=[{key:"transcribe",label:"Transcribe audio",detail:"Finding words + language switches"},{key:"speakers",label:"Map speakers",detail:"Separating voices into a speaker graph"},{key:"tasks",label:"Extract tasks",detail:"Turning decisions into accountable work"}],D=document.querySelector("#app");function r(e=""){return String(e).replace(/[&<>'"]/g,s=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[s])}const g=e=>e==null?"—":`${e}%`;function y(){const e=a.result.transcript.map(n=>n.language),s=e.some(n=>n.includes("HI")),t=e.some(n=>n.includes("EN"));return s&&t?"Hindi + English detected":s?"Hindi detected":t?"English detected":""}function B(e){return new Promise(s=>setTimeout(s,e))}function c(){var $;const e=a.result.timings,s=e.total||0,t=!!a.file,n=(($=a.previousResult)==null?void 0:$.confidence)??null,i=a.result.confidence,o=n!==null&&i!==null?i-n:null,d=p=>s>0?Math.round(p/s*100):0,T=a.processing?Math.min(96,(a.activeStage+.45)/u.length*100):100;D.innerHTML=`
    <div class="app-shell">
      <header class="topbar">
        <div class="brand-lockup">
          <div class="brand-mark" aria-hidden="true"><i></i><i></i><b></b></div>
          <div><div class="wordmark">V<span>//</span>ROOM</div><div class="brand-sub">MEETING INTELLIGENCE / 01</div></div>
        </div>
        <div class="topbar-status">
          <span class="status-dot ${a.processing?"is-processing":""}"></span>
          <span>${a.processing?"PIPELINE ACTIVE":"LOCAL PIPELINE READY"}</span>
          <span class="slash">//</span>
          <span class="muted">${r(a.apiBase||location.origin)}</span>
        </div>
        <div class="topbar-right">
          <span class="learned-pill"><span class="mini-spark">✦</span> ${a.learned} learned</span>
          <span class="latency">${s.toFixed(1)}s total</span>
        </div>
      </header>

      <div class="layout-grid">
        <aside class="input-rail">
          <div class="rail-heading"><span class="eyebrow">01 / INPUT</span><span class="rail-state">${t?"READY":"WAITING"}</span></div>
          <h1>Turn room noise<br /><em>into signal.</em></h1>
          <p class="lede">Drop a recording, name the room, and let the pipeline find the work hiding inside.</p>

          <section class="input-section">
            <div class="section-label"><span>Audio source</span><span class="required">required</span></div>
            <div class="dropzone ${t?"has-file":""} ${a.recording?"is-recording":""}" data-dropzone tabindex="0">
              <input id="file-input" type="file" accept="audio/*" hidden />
              <div class="drop-icon" aria-hidden="true">${a.recording?'<span class="record-ring"></span>':"↥"}</div>
              <div class="drop-copy">
                <strong>${a.recording?"Recording from mic…":t?r(a.fileName):"Drop meeting audio here"}</strong>
                <small>${a.recording?"Click stop when you are done":t?"Ready for a fresh pass":"MP3, WAV, M4A · up to 250MB"}</small>
              </div>
              <button class="icon-button" data-action="browse" aria-label="Browse for audio">↗</button>
            </div>
            <div class="audio-actions">
              <button class="subtle-button" data-action="record">${a.recording?"■ Stop recording":"◉ Record from mic"}</button>
              ${t?'<button class="text-button" data-action="clear-file">clear</button>':""}
            </div>
          </section>

          <section class="input-section compact-section">
            <label class="section-label" for="team-name"><span>Team / room name</span><span class="required">required</span></label>
            <input class="field" id="team-name" value="${r(a.teamName)}" placeholder="e.g. Product / India" />
          </section>

          <section class="input-section compact-section">
            <div class="section-label"><span>Who is in the room?</span><span class="required">required</span></div>
            <div class="roster-box">
              <div class="chip-row">${a.roster.map((p,k)=>`<span class="person-chip"><span class="chip-avatar avatar-${k%4}">${r(p.split(" ").map(R=>R[0]).join("").slice(0,2))}</span>${r(p)}<button data-action="remove-roster" data-index="${k}" aria-label="Remove ${r(p)}">×</button></span>`).join("")}</div>
              <input id="roster-input" class="chip-input" placeholder="Add a name + Enter" />
            </div>
          </section>

          <div class="validation-message ${a.validation?"show":""}">${a.validation||"Audio is kept locally so voice corrections can learn from it."}</div>

          <div class="rail-bottom">
            <button class="primary-button process-button" data-action="process" ${a.processing?"disabled":""}>
              <span>${a.processing?"Processing…":a.hasRerun?"Re-run same recording":"Process meeting"}</span><b>↗</b>
            </button>
            <div class="mode-panel">
              <div class="mode-row"><span class="tiny-label">API</span><span class="switch-label">FastAPI</span></div>
              <input id="api-base" class="field field-small" value="${r(a.apiBase)}" aria-label="FastAPI base URL" />
            </div>
          </div>
        </aside>

        <main class="results-canvas">
          <div class="canvas-topline">
            <div><span class="eyebrow">02 / OUTPUT</span><h2>${a.processing?"The room is becoming legible.":"Meeting signal, sorted."}</h2></div>
            <div class="output-actions"><span class="run-label">RUN <strong>${r(a.result.runId)}</strong></span><button class="ghost-button ${a.transcriptOpen?"active":""}" data-action="toggle-transcript">${a.transcriptOpen?"Hide":"Show"} transcript</button></div>
          </div>
          ${a.error?`<div class="error-banner"><span class="error-icon">!</span><div><strong>${r(a.error.title||"Backend unavailable")}</strong><span>${r(a.error.message||a.error)}</span></div><button data-action="dismiss-error">×</button></div>`:""}
          <div class="progress-strip ${a.processing?"processing":""}"><span style="width:${T}%"></span></div>
          ${a.processing?U():V()}
        </main>

        <aside class="signal-rail">
          <div class="rail-heading"><span class="eyebrow">03 / SIGNAL</span><span class="rail-state">LIVE</span></div>
          <section class="signal-card confidence-card">
            <div class="card-kicker"><span class="accent-dot"></span> SPEAKER MAP CONFIDENCE</div>
            <div class="confidence-value">${i??"—"}<small>${i===null?"":"%"}</small></div>
            <div class="confidence-bar"><span style="width:${i??0}%"></span></div>
            <div class="confidence-foot"><span>${i===null?"no run yet":i>=80?"high signal":"needs review"}</span><span>${a.hasRerun&&o!==null?`${o>=0?"+":""}${o.toFixed(1)}% since correction`:"baseline run"}</span></div>
          </section>
          <section class="signal-card timings-card">
            <div class="card-kicker"><span class="accent-dot dim"></span> TIMINGS / ${s.toFixed(1)}s TOTAL</div>
            <div class="timing-list">
              ${v("01","Transcribe",e.transcribe,d(e.transcribe),"lime")}
              ${v("02","Map speakers",e.speakers,d(e.speakers),"violet")}
              ${v("03","Extract tasks",e.tasks,d(e.tasks),"amber")}
            </div>
          </section>
          <section class="signal-card learning-card">
            <div class="card-kicker"><span class="accent-dot"></span> CORRECTION MEMORY</div>
            <div class="learning-head"><strong>${a.learned===0?"Ready to learn.":`${a.learned} correction${a.learned===1?"":"s"} learned.`}</strong><span>✦</span></div>
            <p>${a.learned===0?"Edit a task, speaker, or note. VoiceRoom will carry that signal into the next pass.":"The next pass will use these edits to tighten the speaker map and task ownership."}</p>
            <div class="learning-compare"><span>before</span><b>${g(n)}</b><i></i><span>after</span><b class="after-value">${a.hasRerun?g(i):"—"}</b></div>
            ${a.hasRerun?'<button class="text-button compare-button" data-action="toggle-compare">view before / after →</button>':""}
          </section>
          <section class="signal-card glossary-card">
            <div class="card-kicker"><span class="accent-dot dim"></span> GLOSSARY</div>
            <p class="small-copy">Names and terms to keep intact.</p>
            <div class="glossary-list">${a.glossary.map(p=>`<span>${r(p)}</span>`).join("")}</div>
            <form id="glossary-form" class="glossary-form"><input name="term" placeholder="Add term" aria-label="Add glossary term" required /><button aria-label="Add glossary term">+</button></form>
          </section>
        </aside>
      </div>
      ${a.toast?`<div class="toast" role="status"><span>✦</span>${r(a.toast)}</div>`:""}
      <footer class="footer-line"><span>VOICE + LANGUAGE + ACCOUNTABILITY</span><span>built for the room, not the feed</span><span>v0.4 / ${new Date().getFullYear()}</span></footer>
    </div>
  `}function v(e,s,t,n,i){return`<div class="timing-row"><div class="timing-meta"><span>${e}</span><strong>${s}</strong><b>${t.toFixed(1)}s</b></div><div class="timing-bar"><i class="bar-${i}" style="width:${n}%"></i></div></div>`}function U(){return`<section class="processing-view"><div class="process-orbit"><span></span><span></span><span></span><b>V//R</b></div><div class="processing-copy"><span class="eyebrow">PIPELINE / ${String(a.activeStage+1).padStart(2,"0")} OF 03</span><h3>${r(u[Math.min(a.activeStage,2)].label)}</h3><p>${r(u[Math.min(a.activeStage,2)].detail)}<span class="ellipsis">...</span></p></div><div class="stage-list">${u.map((e,s)=>`<div class="stage-row ${s<a.activeStage?"done":""} ${s===a.activeStage?"active":""}"><span class="stage-number">${s<a.activeStage?"✓":`0${s+1}`}</span><div><strong>${e.label}</strong><small>${s<a.activeStage?"complete":s===a.activeStage?"working now":"queued"}</small></div><span class="stage-mark">${s<a.activeStage?"●":s===a.activeStage?"◌":"○"}</span></div>`).join("")}</div></section>`}function V(){return`<section class="results-stack">
    <div class="result-meta-row"><span><b>${a.result.people.length}</b> people / <b>${a.result.people.reduce((e,s)=>e+s.tasks.length,0)}</b> tasks found</span>${y()?`<span class="language-badge"><i></i> ${y()}</span>`:""}</div>
    ${a.result.people.length?"":'<p class="lede">Process a meeting to see per-person tasks and notes here.</p>'}
    ${a.compareOpen?W():""}
    <div class="people-grid">${a.result.people.map(z).join("")}</div>
    ${a.transcriptOpen?H():""}
  </section>`}function z(e){const s=e.tasks.length;return`<article class="person-card accent-${e.color}">
    <div class="person-head"><div class="person-id"><span class="person-avatar">${r(e.initials)}</span><div><h3>${r(e.name)}</h3><span>${s} task${s===1?"":"s"} / owner view</span></div></div><button class="voice-button" data-action="read-tasks" data-person-id="${r(e.id)}"><span>◖</span> Read my tasks</button></div>
    <div class="notes-block"><span class="mini-label">NOTES <em>editable</em></span><textarea data-note-person="${r(e.id)}" aria-label="Notes for ${r(e.name)}">${r(e.notes)}</textarea></div>
    <div class="tasks-block"><div class="task-header"><span class="mini-label">TASKS / ${String(s).padStart(2,"0")}</span><span class="task-hint">click any field to edit</span></div>
      ${e.tasks.map(t=>G(e,t)).join("")}
    </div>
  </article>`}function G(e,s){const t=b([...a.roster,"Meeting"]).map(n=>`<option ${n===s.assignee?"selected":""}>${r(n)}</option>`).join("");return`<div class="task-row" data-task-row="${r(s.id)}"><span class="task-check">□</span><div class="task-fields"><input class="task-text" data-task-field="text" data-person-id="${r(e.id)}" data-task-id="${r(s.id)}" value="${r(s.text)}" aria-label="Task text" /><div class="task-detail-fields"><label>by <input data-task-field="assigner" data-person-id="${r(e.id)}" data-task-id="${r(s.id)}" value="${r(s.assigner)}" aria-label="Task assigner" /></label><label>to <select data-task-field="assignee" data-person-id="${r(e.id)}" data-task-id="${r(s.id)}" aria-label="Task assignee">${t}</select></label><label>due <input data-task-field="deadline" data-person-id="${r(e.id)}" data-task-id="${r(s.id)}" value="${r(s.deadline)}" aria-label="Task deadline" /></label></div></div></div>`}function H(){return`<section class="transcript-panel"><div class="panel-heading"><div><span class="eyebrow">TRANSCRIPT / CODE-SWITCH TRACE</span><h3>What the room actually said</h3></div><span class="transcript-note"><i></i> language mix visible</span></div><div class="transcript-list">${a.result.transcript.map(e=>`<div class="transcript-line"><time>${r(e.timestamp)}</time><select data-transcript-speaker="${r(e.id)}" aria-label="Speaker label for ${r(e.timestamp)}">${b(a.roster).map(s=>`<option ${s===e.speaker?"selected":""}>${r(s)}</option>`).join("")}<option ${a.roster.includes(e.speaker)?"":"selected"}>${r(e.speaker)}</option></select><span class="language-tag ${e.language.includes("HI")?"mixed":""}">${r(e.language)}</span><p>${Y(e.text)}</p></div>`).join("")}</div></section>`}function Y(e){return r(e).replace(/(Toh|Haan|Theek hai)/g,"<mark>$1</mark>")}function W(){var t;const e=((t=a.previousResult)==null?void 0:t.confidence)??null,s=a.result.confidence;return`<div class="comparison-panel"><div><span class="eyebrow">BEFORE / AFTER</span><strong>Correction loop closed.</strong><p>Speaker map and ownership improved after ${a.learned} learned correction${a.learned===1?"":"s"}.</p></div><div class="compare-score"><span>${g(e)}</span><b>→</b><strong>${g(s)}</strong></div><button data-action="toggle-compare" class="text-button">close ×</button></div>`}function b(e){return[...new Set(e.filter(Boolean))]}function J(){var e;return a.teamName=((e=document.querySelector("#team-name"))==null?void 0:e.value.trim())||"",a.file?a.teamName?a.roster.length<1?"Add at least one person to the roster.":"":"Name the team or room so the summary has context.":"Add an audio file or record the meeting first."}async function K(){if(a.validation=J(),a.validation){c();return}const e=a.hasRerun||a.learned>0;a.processing=!0,a.error="",a.activeStage=0,c();const s=O({file:a.file,teamName:a.teamName,roster:a.roster}),t=s.then(()=>!0,()=>!0);let n=!1;t.then(()=>{n=!0});try{for(let o=0;o<u.length&&!n;o+=1)a.activeStage=o,c(),await Promise.race([B(1500),t]);const i=await s;a.previousResult=a.result.people.length?a.result:null,a.result=i,a.hasRerun=e,a.activeStage=3,l(e?"Fresh pass complete. The correction is visible.":"Meeting processed. Review the signal below.")}catch(i){const o=a.result.people.length?" Your previous result is still shown.":"";a.error={title:"Processing failed",message:`${i.message}${o}`},a.activeStage=3}finally{a.processing=!1,c()}}function l(e){a.toast=e,c(),window.clearTimeout(l.timer),l.timer=window.setTimeout(()=>{a.toast="",c()},3200)}async function S(e){const s=a.result.people.find(o=>o.id===e.dataset.personId),t=s==null?void 0:s.tasks.find(o=>o.id===e.dataset.taskId);if(!t)return;const n=e.value.trim();if(!n||t[e.dataset.taskField]===n)return;const i={...t};t[e.dataset.taskField]=n;try{await L({teamName:a.teamName,before:i,after:{...t}}),a.learned+=1,l("Correction learned. Next pass will use it.")}catch(o){a.error={title:"Correction not saved",message:`${o.message}. The edit is only on this page.`},c()}}function Q(e){const s=a.result.people.find(t=>t.id===e.dataset.notePerson);!s||s.notes===e.value||(s.notes=e.value,l("Note updated on this page. Notes are not used for learning."))}async function X(e){const s=a.result.transcript.find(n=>n.id===e.dataset.transcriptSpeaker);if(!s||s.speaker===e.value)return;const t=s.speaker;s.speaker=e.value;try{const n=await P({teamName:a.teamName,runId:a.result.runId,line:s,from:t,to:s.speaker});a.learned+=1,l(n.voice_learned?"Speaker label learned. Voice updated.":"Speaker label saved. Voice not updated (voice embeddings unavailable).")}catch(n){a.error={title:"Speaker fix not saved",message:`${n.message}. The label is only changed on this page.`},c()}}function E(e){e&&(a.file=e,a.fileName=e.name,a.validation="",c(),l(`${e.name} is ready for processing.`))}async function Z(){var e;if(a.recording&&a.mediaRecorder){a.mediaRecorder.stop();return}if(!((e=navigator.mediaDevices)!=null&&e.getUserMedia)||!window.MediaRecorder){a.error={title:"Microphone recording is not available",message:"Drop an audio file instead, or open this preview in a browser with microphone permissions."},c();return}try{const s=await navigator.mediaDevices.getUserMedia({audio:!0}),t=new MediaRecorder(s);a.chunks=[],a.mediaRecorder=t,t.ondataavailable=n=>a.chunks.push(n.data),t.onstop=()=>{const n=new Blob(a.chunks,{type:"audio/webm"});a.file=new File([n],"voice-room-recording.webm",{type:"audio/webm"}),a.fileName="voice-room-recording.webm",a.recording=!1,s.getTracks().forEach(i=>i.stop()),c(),l("Recording captured. Ready for processing.")},t.start(),a.recording=!0,c()}catch{a.error={title:"Microphone permission was not granted",message:"You can still drop an audio file to continue."},c()}}async function ee(e){const s=a.result.people.find(n=>n.id===e);if(!s)return;const t=s.tasks.length?`${s.name}'s tasks. ${s.tasks.map(n=>`${n.text}, due ${n.deadline}`).join(". ")}.`:`${s.name} has no tasks.`;try{const n=await M(t),i=new Audio(n);i.onended=()=>URL.revokeObjectURL(n),await i.play(),l("Playing ElevenLabs task readout.");return}catch(n){if("speechSynthesis"in window){window.speechSynthesis.cancel();const i=new SpeechSynthesisUtterance(t);i.rate=.94,window.speechSynthesis.speak(i),l(`ElevenLabs unavailable (${n.message}). Playing the browser voice.`)}else a.error={title:"Voice playback failed",message:n.message},c()}}async function ae(e){var i;const s=new FormData(e),t=String(s.get("term")||"").trim();if(!t)return;const n=((i=document.querySelector("#team-name"))==null?void 0:i.value.trim())||a.teamName;if(!n){a.validation="Name the team or room first so the term is saved for it.",c();return}try{await x({teamName:n,term:t}),a.glossary=b([...a.glossary,t]),e.reset(),l(`${t} added to the glossary.`)}catch(o){a.error={title:"Glossary term not saved",message:o.message},c()}}document.addEventListener("click",async e=>{var n;const s=e.target.closest("[data-action]");if(!s)return;const t=s.dataset.action;t==="browse"&&((n=document.querySelector("#file-input"))==null||n.click()),t==="record"&&await Z(),t==="clear-file"&&(a.file=null,a.fileName="",c()),t==="process"&&await K(),t==="remove-roster"&&(a.roster.splice(Number(s.dataset.index),1),c()),t==="toggle-transcript"&&(a.transcriptOpen=!a.transcriptOpen,c()),t==="toggle-compare"&&(a.compareOpen=!a.compareOpen,c()),t==="dismiss-error"&&(a.error="",c()),t==="read-tasks"&&await ee(s.dataset.personId)});document.addEventListener("change",async e=>{var t;const s=e.target;s.matches("#file-input")&&E((t=s.files)==null?void 0:t[0]),s.matches("[data-task-field]")&&await S(s),s.matches("[data-transcript-speaker]")&&await X(s)});document.addEventListener("focusout",async e=>{const s=e.target;s.matches("[data-task-field]")&&await S(s),s.matches("[data-note-person]")&&Q(s),s.matches("#api-base")&&(A(s.value),a.apiBase=m())});document.addEventListener("keydown",e=>{var s;if(e.key==="Enter"&&e.target.matches("#roster-input")){e.preventDefault();const t=e.target.value.trim();t&&!a.roster.includes(t)&&(a.roster.push(t),e.target.value="",c(),(s=document.querySelector("#roster-input"))==null||s.focus())}});document.addEventListener("submit",async e=>{e.target.matches("#glossary-form")&&(e.preventDefault(),await ae(e.target))});document.addEventListener("dragover",e=>{e.target.closest("[data-dropzone]")&&(e.preventDefault(),e.target.closest("[data-dropzone]").classList.add("is-dragging"))});document.addEventListener("dragleave",e=>{var s;(s=e.target.closest("[data-dropzone]"))==null||s.classList.remove("is-dragging")});document.addEventListener("drop",e=>{var t;const s=e.target.closest("[data-dropzone]");s&&(e.preventDefault(),s.classList.remove("is-dragging"),E((t=e.dataTransfer.files)==null?void 0:t[0]))});document.addEventListener("click",e=>{var s;e.target.closest("[data-dropzone]")&&!e.target.closest("button")&&((s=document.querySelector("#file-input"))==null||s.click())});c();
