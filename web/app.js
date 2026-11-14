// DOM wiring. Logic lives in lib.js. Element ids are listed in PLAN.md.
import {
  PROTOCOL, TYPING_IDLE_MS, partialMsg, finalMsg, tapMsg, parseFrame, parseServerMessage,
  splitKeyword, backoffMs, isWholeInsert, makeFrameGate, makeLatencyTracker, makeTranscriber,
} from './lib.js';

const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), connection: $('connection'), banner: $('mode-banner'), words: $('words'),
  changes: $('changes'), form: $('command-form'), command: $('command'), mic: $('mic'),
  ptt: $('ptt'), undo: $('undo'), latency: $('latency'), timing: $('timing'), error: $('error'),
};
const ctx = el.video.getContext('2d');
const wsBase = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
const latency = makeLatencyTracker();
const PROXY_NOTE = 'command sent → first frame tagged with its chunk drawn to the canvas (page clock; chunk-tag proxy, not proof the picture changed or was painted)';

function showError(text) {
  el.error.textContent = text || '';
  el.error.hidden = !text;
}

// ---------- control socket ----------
let control = null;
let attempt = 0;

function send(msg) {
  if (control?.readyState !== WebSocket.OPEN) {
    showError(`Not sent (disconnected): ${msg.text ?? msg.action}`); // dropped, never replayed
    return false;
  }
  control.send(JSON.stringify(msg));
  if (msg.type === 'final') latency.sent(msg.t_client);
  return true;
}

function setConnected(open) {
  el.connection.dataset.state = open ? 'open' : control ? 'closed' : 'connecting';
  el.connection.textContent = open ? 'Connected' : 'Connection lost — reconnecting…';
  el.changes.dataset.stale = String(!open);
  el.undo.disabled = !open;
  for (const b of el.changes.querySelectorAll('button')) b.disabled = !open;
}

function connectControl() {
  const ws = new WebSocket(wsBase + PROTOCOL.controlPath);
  control = ws;
  ws.onopen = () => {
    attempt = 0;
    setConnected(true);
    showError('');
  };
  ws.onmessage = (e) => {
    try {
      onServer(e.data);
    } catch (err) {
      showError(`Display error: ${err.message}`);
    }
  };
  ws.onclose = () => {
    if (control !== ws) return;
    latency.reset(); // a stale command can no longer be correlated
    setConnected(false);
    setTimeout(connectControl, backoffMs(attempt++));
  };
}

function onServer(raw) {
  const m = typeof raw === 'string' ? parseServerMessage(raw) : null;
  if (!m) return showError('Ignored an invalid message from the server');
  if (m.error) showError(`Server: ${m.error}`);
  if (m.fake !== undefined) el.banner.hidden = !m.fake;
  if (m.changes) renderChanges(m.changes);
  if (m.heard !== undefined) renderWords(m);
  latency.onServer(m);
}

function renderWords({ heard, keyword, unmatched, kind }) {
  el.words.replaceChildren(
    ...splitKeyword(heard, keyword).map((p) => {
      if (!p.mark) return document.createTextNode(p.text);
      const mark = document.createElement('mark');
      mark.textContent = p.text;
      return mark;
    }),
  );
  el.words.dataset.unmatched = String(Boolean(unmatched));
  el.words.dataset.kind = kind || '';
}

function renderChanges(changes) {
  const open = control?.readyState === WebSocket.OPEN;
  el.changes.replaceChildren(
    ...changes.map((c) => {
      const li = document.createElement('li');
      li.className = 'chip';
      li.dataset.id = c.id;
      const label = document.createElement('span');
      label.className = 'chip-label';
      label.textContent = c.label;
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'chip-remove';
      x.textContent = '×';
      x.setAttribute('aria-label', `Remove ${c.label}`);
      x.disabled = !open;
      x.addEventListener('click', () => send(tapMsg('remove', c.id))); // chips change only when the server says so
      li.append(label, x);
      return li;
    }),
  );
}

// ---------- frames socket ----------
const gate = makeFrameGate();
let frameAttempt = 0;

function connectFrames() {
  const ws = new WebSocket(wsBase + PROTOCOL.framesPath);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => (frameAttempt = 0);
  ws.onmessage = async (e) => {
    const f = parseFrame(e.data);
    if (!f) return;
    const seq = gate.next();
    let bmp;
    try {
      bmp = await createImageBitmap(new Blob([f.jpeg], { type: 'image/jpeg' }));
    } catch {
      return; // undecodable frame: skip it, keep the last good one
    }
    if (!gate.tryDraw(seq)) return bmp.close(); // a newer frame was already drawn
    if (el.video.width !== bmp.width) el.video.width = bmp.width;
    if (el.video.height !== bmp.height) el.video.height = bmp.height;
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    const r = latency.onFrameDrawn(f.chunk);
    if (r) renderLatency(r);
  };
  ws.onclose = () => setTimeout(connectFrames, backoffMs(frameAttempt++));
}

function renderLatency({ ms, stamps }) {
  el.latency.textContent = `${ms} ms`;
  const s = stamps ? ` · server stamps (ms after heard): ${Object.entries(stamps).map(([k, v]) => `${k} ${v}`).join(', ')}` : '';
  el.timing.textContent = PROXY_NOTE + s;
}

// ---------- text box (Wispr Flow pastes here) ----------
let idleTimer = 0;

function sendFinalFromBox() {
  clearTimeout(idleTimer);
  const text = el.command.value.trim();
  el.command.value = '';
  if (text) send(finalMsg(text));
}

el.command.addEventListener('input', (e) => {
  clearTimeout(idleTimer);
  if (isWholeInsert(e.inputType, e.data)) return sendFinalFromBox();
  const text = el.command.value.trim();
  if (!text) return;
  send(partialMsg(text));
  idleTimer = setTimeout(sendFinalFromBox, TYPING_IDLE_MS);
});
el.form.addEventListener('submit', (e) => {
  e.preventDefault();
  sendFinalFromBox();
});
// Keep the box focused so Wispr Flow always has a target, except while using another control.
document.addEventListener('pointerup', () => setTimeout(() => el.command.focus(), 0));

el.undo.addEventListener('click', () => send(tapMsg('undo')));

// ---------- live mic ----------
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const transcriber = makeTranscriber(send);
let rec = null;
let listening = null; // 'toggle' | 'ptt' | null

function startListening(mode) {
  if (!Recognition || listening) return;
  listening = mode;
  rec = new Recognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = 'en-US';
  transcriber.start();
  rec.onresult = (e) => transcriber.handle(Array.from(e.results, (r) => ({ transcript: r[0].transcript, isFinal: r.isFinal })));
  rec.onerror = (e) => e.error !== 'no-speech' && e.error !== 'aborted' && showError(`Mic: ${e.error}`);
  rec.onend = () => {
    transcriber.flush(); // leftover interim text becomes one final, once
    rec = null;
    listening = null;
    renderMic();
  };
  rec.start();
  renderMic();
}

function stopListening(mode) {
  if (rec && listening === mode) rec.stop(); // onend flushes
}

function renderMic() {
  el.mic.setAttribute('aria-pressed', String(listening === 'toggle'));
  el.ptt.setAttribute('aria-pressed', String(listening === 'ptt'));
}

if (!Recognition) {
  for (const b of [el.mic, el.ptt]) {
    b.disabled = true;
    b.title = 'This browser has no speech recognition; use Chrome';
  }
} else {
  el.mic.addEventListener('click', () => (listening === 'toggle' ? stopListening('toggle') : startListening('toggle')));
  el.ptt.addEventListener('pointerdown', (e) => {
    el.ptt.setPointerCapture(e.pointerId);
    startListening('ptt');
  });
  for (const ev of ['pointerup', 'pointercancel']) el.ptt.addEventListener(ev, () => stopListening('ptt'));
  // Backtick is push-to-talk: it never types into the box.
  document.addEventListener('keydown', (e) => {
    if (e.key !== '`') return;
    e.preventDefault();
    if (!e.repeat) startListening('ptt');
  });
  document.addEventListener('keyup', (e) => e.key === '`' && stopListening('ptt'));
  window.addEventListener('blur', () => stopListening('ptt'));
}

setConnected(false);
el.connection.dataset.state = 'connecting';
el.connection.textContent = 'Connecting…';
el.latency.textContent = '—';
el.timing.textContent = 'Delay unavailable until the server correlates a command with a chunk.';
connectControl();
connectFrames();
el.command.focus();
