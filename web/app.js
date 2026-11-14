// DOM wiring. Logic lives in lib.js. Element ids are listed in PLAN.md.
import {
  PROTOCOL, TYPING_IDLE_MS, BURST_IDLE_MS, partialMsg, finalMsg, tapMsg, parseFrame, parseServerMessage,
  splitKeyword, backoffMs, makeFrameGate, makeLatencyTracker, makeTranscriber,
  changesKey, commandKey, makeInputClassifier, PUSH_TO_TALK_CODE, isPushToTalk, micErrorText,
  wsBaseUrl, fitSize,
} from './lib.js';

const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), connection: $('connection'), banner: $('mode-banner'), words: $('words'),
  changes: $('changes'), form: $('command-form'), command: $('command'), mic: $('mic'),
  ptt: $('ptt'), undo: $('undo'), latency: $('latency'), timing: $('timing'), error: $('error'),
};
const ctx = el.video.getContext('2d');
// Same origin by default; `?server=host:port` points the page at another server (e.g. the engine on :8000).
const params = new URLSearchParams(location.search);
// `?server=` also takes a full ws:// / wss:// / https:// URL, e.g. an HTTPS tunnel to the engine.
const wsBase = wsBaseUrl(params.get('server'), location);
// `?debug=1` logs every text-box input to the console, to see how Wispr Flow inserts on this laptop.
const DEBUG = params.get('debug') === '1';
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
  // Show the words at once; the server's version (keyword, red if unmatched) replaces them when it lands.
  if (msg.type === 'partial' || msg.type === 'final') renderLocalWords(msg.text, msg.type);
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
    el.latency.textContent = '—'; // no number without a measured frame
    setConnected(false);
    setTimeout(connectControl, backoffMs(attempt++));
  };
}

// State arrives every chunk; only touch the DOM when what it shows has changed.
let shownChanges = null;
let shownCommand = null;

function onServer(raw) {
  const m = typeof raw === 'string' ? parseServerMessage(raw) : null;
  if (!m) return showError('Ignored an invalid message from the server');
  if (m.error) return showError(`Server: ${m.error}`); // an error reply carries no state
  el.banner.hidden = m.fake !== true; // the real engine does not send `fake`
  if (m.changes && changesKey(m.changes) !== shownChanges) {
    shownChanges = changesKey(m.changes);
    renderChanges(m.changes);
  }
  if (m.heard !== undefined && commandKey(m) !== shownCommand) {
    shownCommand = commandKey(m);
    renderWords(m);
  }
  latency.onServer(m);
}

function renderLocalWords(text, kind) {
  el.words.replaceChildren(document.createTextNode(text));
  el.words.dataset.unmatched = 'false';
  el.words.dataset.kind = kind;
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
  el.words.dataset.kind = kind || 'server';
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
    // Size the canvas to the screen space it has (in device pixels) at the frame's own aspect ratio,
    // read from the decoded frame, then upscale with high-quality smoothing.
    const size = fitSize(bmp.width, bmp.height, el.video.clientWidth, el.video.clientHeight, window.devicePixelRatio || 1);
    if (el.video.width !== size.width || el.video.height !== size.height) {
      el.video.width = size.width; // resizing a canvas resets its drawing settings
      el.video.height = size.height;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, size.width, size.height);
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
const inputs = makeInputClassifier();

function sendFinalFromBox() {
  clearTimeout(idleTimer);
  inputs.reset();
  const text = el.command.value.trim();
  el.command.value = '';
  if (text) send(finalMsg(text));
}

// No Enter needed: a paste is final at once, typing becomes final after a pause.
el.command.addEventListener('input', (e) => {
  clearTimeout(idleTimer);
  if (e.isComposing) return; // mid-composition (accents, IME): wait for the composed text
  const kind = inputs.classify(e.inputType, e.data, performance.now());
  if (DEBUG) console.info(`[web] input ${e.inputType} chars=${e.data?.length ?? '-'} -> ${kind}`);
  if (kind === 'whole') return sendFinalFromBox();
  const text = el.command.value.trim();
  if (!text) return;
  send(partialMsg(text));
  idleTimer = setTimeout(sendFinalFromBox, kind === 'burst' ? BURST_IDLE_MS : TYPING_IDLE_MS);
});
el.form.addEventListener('submit', (e) => {
  e.preventDefault();
  sendFinalFromBox();
});
// Keep the box focused at all times so a Wispr Flow paste always lands in it.
const refocus = () => setTimeout(() => document.activeElement !== el.command && el.command.focus(), 0);
// Buttons never take focus: blocking mousedown keeps focus in the box, and the click still fires.
document.addEventListener('mousedown', (e) => e.target.closest('button') && e.preventDefault());
// Anything else on the page that takes focus gives it straight back. When focus leaves the page
// (address bar, another app) do not fight for it: the window focus handler restores it on return.
el.command.addEventListener('blur', () => document.hasFocus() && refocus());
window.addEventListener('focus', refocus); // returning from another app
document.addEventListener('pointerup', refocus);

el.undo.addEventListener('click', () => send(tapMsg('undo')));

// ---------- live mic ----------
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const transcriber = makeTranscriber(send);
let rec = null;
let listening = null; // what the user asked for: 'toggle' | 'ptt' | null
let startedAt = 0;

function startListening(mode) {
  if (!Recognition || listening) return;
  listening = mode;
  begin();
  renderMic();
}

function begin() {
  rec = new Recognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = 'en-US';
  transcriber.start();
  startedAt = performance.now();
  rec.onresult = (e) => transcriber.handle(Array.from(e.results, (r) => ({ transcript: r[0].transcript, isFinal: r.isFinal })));
  rec.onerror = (e) => {
    const text = micErrorText(e.error);
    if (!text) return;
    showError(text);
    listening = null; // a real failure: do not restart into it
  };
  rec.onend = () => {
    transcriber.flush(); // leftover interim text becomes one final, once
    rec = null;
    // Chrome ends recognition by itself after a pause; keep listening while the mic toggle is on,
    // unless it ended at once (then restarting would just spin).
    if (listening === 'toggle' && performance.now() - startedAt > 500) return begin();
    listening = null;
    renderMic();
  };
  rec.start();
}

function stopListening(mode) {
  if (listening !== mode) return;
  listening = null; // so onend does not restart
  renderMic();
  rec?.stop(); // onend flushes the last words
}

function renderMic() {
  el.mic.setAttribute('aria-pressed', String(listening === 'toggle'));
  el.ptt.setAttribute('aria-pressed', String(listening === 'ptt'));
}

const micUnavailable = !Recognition
  ? 'This browser has no speech recognition; use Chrome'
  : !window.isSecureContext
    ? 'The mic needs HTTPS or localhost; open the page through an HTTPS tunnel or an SSH port-forward'
    : null;
const keyLabel = { Backquote: '`' }[PUSH_TO_TALK_CODE] ?? PUSH_TO_TALK_CODE;

if (micUnavailable) {
  for (const b of [el.mic, el.ptt]) {
    b.disabled = true;
    b.title = micUnavailable;
  }
} else {
  el.ptt.title = `Hold to talk (or hold the ${keyLabel} key)`;
  el.mic.addEventListener('click', () => (listening === 'toggle' ? stopListening('toggle') : startListening('toggle')));
  el.ptt.addEventListener('pointerdown', (e) => {
    el.ptt.setPointerCapture(e.pointerId);
    startListening('ptt');
  });
  for (const ev of ['pointerup', 'pointercancel']) el.ptt.addEventListener(ev, () => stopListening('ptt'));
  // Push-to-talk key (PUSH_TO_TALK_CODE in lib.js): never with Ctrl, which is Wispr Flow's hotkey,
  // and it never types into the box.
  document.addEventListener('keydown', (e) => {
    if (!isPushToTalk(e)) return;
    e.preventDefault();
    if (!e.repeat) startListening('ptt');
  });
  document.addEventListener('keyup', (e) => e.code === PUSH_TO_TALK_CODE && stopListening('ptt'));
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
