// DOM wiring. Logic lives in lib.js. Element ids are listed in PLAN.md.
import {
  PROTOCOL, finalizeDelay, partialMsg, finalMsg, tapMsg, parseFrame, parseServerMessage,
  backoffMs, makeFrameGate, makeLatencyTracker, makeTranscriber,
  changesKey, commandKey, makeInputClassifier, wordsLabels, makeHeardTracker, PUSH_TO_TALK_CODE, isPushToTalk, micErrorText,
  wsBaseUrl, fitSize, videoNotice, formatStamps, isResetShortcut, makeOrderedFinals, makeSpeechEndpointer,
  makeTypeBox, TYPE_REVEAL_MS,
} from './lib.js';

const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), connection: $('connection'), banner: $('mode-banner'), words: $('words'),
  changes: $('changes'), form: $('command-form'), command: $('command'), keyboard: $('keyboard'),
  ptt: $('ptt'), undo: $('undo'), latency: $('latency'), timing: $('timing'), error: $('error'),
  notice: $('video-notice'), reset: $('reset'), frame: $('frame'), fullscreen: $('fullscreen'),
  pttRing: $('ptt-ring'),
};
const ctx = el.video.getContext('2d');
// Same origin by default; `?server=host:port` points the page at another server (e.g. the engine on :8000).
const params = new URLSearchParams(location.search);
// `?server=` also takes a full ws:// / wss:// / https:// URL, e.g. an HTTPS tunnel to the engine.
const wsBase = wsBaseUrl(params.get('server'), location);
// `?debug=1` logs every text-box input to the console, to see how Wispr Flow inserts on this laptop.
const DEBUG = params.get('debug') === '1';
const latency = makeLatencyTracker();
const heardTracker = makeHeardTracker();
const PROXY_NOTE = 'command sent → first frame tagged with its chunk drawn to the canvas (page clock; chunk-tag proxy, not proof the picture changed or was painted)';
el.latency.title = PROXY_NOTE;

// What the notice over the video says: connecting, connection lost, or paused (no frames).
const link = { controlOpen: false, framesOpen: false, lastFrameAt: null, everConnected: false };
function renderNotice() {
  const text = videoNotice({ ...link, now: performance.now() });
  el.notice.textContent = text || '';
  el.notice.hidden = !text;
}
setInterval(renderNotice, 250);

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
  if (msg.type === 'final') heardTracker.sentFinal(msg.text, msg.t_client);
  if (msg.type === 'partial') heardTracker.sentPartial(msg.text);
  // Show the words at once; the server's version (keyword, red if unmatched) replaces them when it lands.
  if (msg.type === 'partial' || msg.type === 'final') renderLocalWords(msg.text, msg.type);
  return true;
}

// Status capsule: the delay while connected ("Live" until a command is measured); Reconnecting… while
// retrying (Connecting… before the first connection); Offline once several retries in a row have
// failed. The notice under the header says more.
const OFFLINE_AFTER = 5; // failed attempts in a row (about 8 s with the backoff)
let connState = 'connecting';
let delayMs = null;
function renderStatus() {
  el.connection.dataset.state = connState;
  const text = connState === 'open'
    ? delayMs === null ? 'Live' : `${(delayMs / 1000).toFixed(1)} s`
    : connState === 'offline' ? 'Offline' : link.everConnected ? 'Reconnecting…' : 'Connecting…';
  el.latency.textContent = text;
  el.latency.setAttribute('aria-label', connState === 'open' && delayMs !== null ? `Live, ${text} from command to video` : text);
  // While connected the capsule warns about a slow delay: amber over 1.5 s, red over 3 s.
  el.connection.dataset.level = delayMs === null ? '' : delayMs > 3000 ? 'bad' : delayMs > 1500 ? 'slow' : 'ok';
}
function setConnected(open) {
  connState = open ? 'open' : attempt >= OFFLINE_AFTER ? 'offline' : 'connecting';
  renderStatus();
  el.changes.dataset.stale = String(!open);
  el.undo.disabled = !open;
  el.reset.disabled = !open;
  for (const b of el.changes.querySelectorAll('button')) b.disabled = !open;
}

function connectControl() {
  const ws = new WebSocket(wsBase + PROTOCOL.controlPath);
  control = ws;
  ws.onopen = () => {
    attempt = 0;
    link.controlOpen = true;
    link.everConnected ||= link.framesOpen; // "lost" only after both sockets have worked once
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
    link.controlOpen = false;
    renderNotice();
    latency.reset(); // a stale command can no longer be correlated
    ordered.dropAll(); // held spoken commands are dropped, never replayed after a reconnect
    showDelay(null); // no number without a measured frame
    setTimeout(connectControl, backoffMs(attempt++));
    setConnected(false);
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
    const status = heardTracker.classify(m.heard, m.t_client);
    // Our own words still being said or typed: the line already shows them (newer than this echo).
    if (status !== 'pending') renderWords({ ...m, confirmed: status === 'confirmed' });
  }
  latency.onServer(m);
}

// The voice line: the words in orange quotes, and before them what is happening. Voice bars while speech
// comes in (from the mic level while push-to-talk is held, a wave for Wispr Flow dictation), nothing
// while typing, a green tick once the server understood the final, an amber "!" when it did not.
const SVG_NS = 'http://www.w3.org/2000/svg';
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const bars = document.createElement('span');
bars.className = 'bars wave';
bars.setAttribute('aria-hidden', 'true');
for (let i = 0; i < 5; i++) bars.append(document.createElement('span'));
let settled = null; // the last finished line, shown again when a draft or a press ends with no words
let lastPartial = '';
let boxKind = 'typing'; // how the text box's words arrived: 'typing', or 'burst' (dictated)

function statusIcon(kind) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('class', 'voice-icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = kind === 'tick'
    ? '<circle cx="8" cy="8" r="8" fill="#34C759"/><path d="M4.6 8.3l2.2 2.2 4.6-4.8" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    : '<circle cx="8" cy="8" r="8" fill="#D97706"/><path d="M8 4.2v4.6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.6" r="1.05" fill="#fff"/>';
  return svg;
}

function span(className, text) {
  const s = document.createElement('span');
  s.className = className;
  if (text) s.textContent = text;
  return s;
}

// state: 'empty' | 'speaking' | 'typing' | 'sent' | 'understood' | 'unmatched' | 'plain'
function renderVoice({ state, text = '', tail = '', mode = null }) {
  el.words.dataset.state = state;
  const parts = [];
  if (mode) {
    bars.classList.toggle('wave', !(mode === 'level' && level.analyser)); // the level loop drives the bars
    parts.push(bars);
  }
  if (state === 'understood') parts.push(statusIcon('tick'));
  if (state === 'unmatched') parts.push(statusIcon('alert'));
  const words = span('voice-text', state === 'empty' ? 'Say something' : text);
  if (tail) words.append(span('voice-tail', tail));
  const body = span('voice-body');
  body.append(span('voice-quote', '“'), words);
  if (state === 'speaking') body.append(span('voice-caret'));
  body.append(span('voice-quote', '”'));
  for (const q of body.querySelectorAll('.voice-quote')) q.setAttribute('aria-hidden', 'true');
  parts.push(body);
  if (state === 'unmatched') parts.push(span('voice-note', ' · Didn’t catch that'));
  el.words.replaceChildren(...parts);
}

// Where words that are still coming in come from: our mic (bars), dictation into the box (wave), or typing.
function partialMode() {
  if (listening === 'ptt') return 'level';
  if (rec) return 'wave'; // a press still finishing
  return boxKind === 'burst' ? 'wave' : null;
}

// Words still coming in. The words two updates agree on are settled; the rest is grey, then a caret.
function showPartial(text) {
  const mode = partialMode();
  if (!mode) {
    lastPartial = '';
    return renderVoice({ state: 'typing', text });
  }
  const a = lastPartial.split(' ');
  const b = text.split(' ');
  let n = 0;
  while (n < a.length - 1 && n < b.length - 1 && a[n] === b[n]) n++;
  lastPartial = text;
  const done = b.slice(0, n).join(' ');
  renderVoice({ state: 'speaking', text: done ? `${done} ` : '', tail: b.slice(n).join(' '), mode });
}

function showSettled() {
  lastPartial = '';
  if (!settled?.heard) return renderVoice({ state: 'empty' });
  const { heard, keyword, unmatched, confirmed } = settled;
  // Same meaning as wordsLabels (lib.js): a tick only for a final the server understood.
  const state = unmatched ? 'unmatched' : wordsLabels({ keyword, unmatched, confirmed }).understood ? 'understood' : 'plain';
  renderVoice({ state, text: heard });
}

// A press or a draft that ended without sending anything shows the last finished line again.
function endUnsent() {
  if (el.words.dataset.state === 'speaking' || el.words.dataset.state === 'typing') showSettled();
}

function renderLocalWords(text, kind) {
  if (kind === 'partial') return showPartial(text);
  lastPartial = '';
  renderVoice({ state: 'sent', text }); // final, until the server says whether it understood
}

function renderWords({ heard, keyword, unmatched, confirmed }) {
  settled = { heard, keyword, unmatched, confirmed };
  showSettled();
}

// Chips are keyed by id: a chip that stays keeps its element (so a click on its × is never lost),
// a new one animates in, and a removed one animates out. They still show only the server's state.
function renderChanges(changes) {
  const open = control?.readyState === WebSocket.OPEN;
  const existing = new Map([...el.changes.querySelectorAll('.chip:not(.leaving)')].map((li) => [li.dataset.id, li]));
  const keep = new Set();
  for (const c of changes) {
    let li = existing.get(c.id);
    if (!li) li = makeChip(c);
    li.querySelector('.chip-label').textContent = c.label;
    li.querySelector('.chip-remove').disabled = !open;
    keep.add(c.id);
    el.changes.append(li); // in the server's order
  }
  for (const [id, li] of existing) {
    if (keep.has(id)) continue;
    li.classList.add('leaving');
    li.querySelector('.chip-remove').disabled = true;
    setTimeout(() => li.remove(), 220);
  }
}

// Chip colours rotate in the order changes first appear; a change brought back by Undo keeps its colour.
const CHIP_COLORS = ['blue', 'red', 'amber'];
const chipColors = new Map(); // change id -> colour
function makeChip(c) {
  const li = document.createElement('li');
  li.className = 'chip';
  li.dataset.id = c.id;
  if (!chipColors.has(c.id)) chipColors.set(c.id, CHIP_COLORS[chipColors.size % CHIP_COLORS.length]);
  li.dataset.color = chipColors.get(c.id);
  const label = document.createElement('span');
  label.className = 'chip-label';
  const x = document.createElement('button');
  x.type = 'button';
  x.className = 'chip-remove';
  x.textContent = '×';
  x.setAttribute('aria-label', `Remove ${c.label}`);
  x.addEventListener('click', () => send(tapMsg('remove', c.id))); // chips change only when the server says so
  li.append(label, x);
  return li;
}

// ---------- frames socket ----------
const gate = makeFrameGate();
let lastFrameSize = null;
// When the space changes with no new frame (window resized, full screen, video paused), re-fit the
// picture's CSS size at once; the next frame redraws it at the new resolution.
new ResizeObserver(() => {
  if (!lastFrameSize) return;
  const size = fitSize(lastFrameSize.w, lastFrameSize.h, el.frame.clientWidth, el.frame.clientHeight, 1);
  el.video.style.width = `${size.width}px`;
  el.video.style.height = `${size.height}px`;
}).observe(el.frame);
let frameAttempt = 0;

function connectFrames() {
  const ws = new WebSocket(wsBase + PROTOCOL.framesPath);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    frameAttempt = 0;
    link.framesOpen = true;
    link.everConnected ||= link.controlOpen;
  };
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
    // Fit the frame (its size read from the decoded image) into the space the stage has, keeping its
    // aspect ratio. The canvas's CSS size is the picture itself, so the rounded corners and shadow sit on
    // it; its backing size is in device pixels and the frame is upscaled with high-quality smoothing.
    const dpr = window.devicePixelRatio || 1;
    lastFrameSize = { w: bmp.width, h: bmp.height };
    const size = fitSize(bmp.width, bmp.height, el.frame.clientWidth, el.frame.clientHeight, dpr);
    if (el.video.width !== size.width || el.video.height !== size.height) {
      el.video.width = size.width; // resizing a canvas resets its drawing settings
      el.video.height = size.height;
      el.video.style.width = `${size.width / dpr}px`;
      el.video.style.height = `${size.height / dpr}px`;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, size.width, size.height);
    bmp.close();
    link.lastFrameAt = performance.now();
    const r = latency.onFrameDrawn(f.chunk);
    if (r) renderLatency(r);
  };
  ws.onclose = () => {
    link.framesOpen = false;
    link.lastFrameAt = null;
    renderNotice();
    setTimeout(connectFrames, backoffMs(frameAttempt++));
  };
}

// Delay: command -> first new frame on screen, in seconds, shown in the status capsule. The server's
// own stage stamps show on hover; the exact definition is in the tooltip.
function showDelay(ms) {
  delayMs = ms;
  renderStatus();
}
function renderLatency({ ms, stamps }) {
  showDelay(ms);
  const s = formatStamps(stamps);
  el.timing.textContent = `${ms} ms · voice → video` + (s ? ` · server: ${s}` : '');
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
  boxKind = kind; // dictated text gets the voice wave on the voice line, typing gets no bars
  revealIfTyping(kind);
  const text = el.command.value.trim();
  if (!text) {
    heardTracker.clearPartials(); // draft deleted: it is no longer pending
    return endUnsent();
  }
  send(partialMsg(text));
  // Sent after a pause, unless it still looks unfinished ("a car,"): then only Enter sends it.
  const delay = finalizeDelay(kind, text);
  if (delay !== null) idleTimer = setTimeout(sendFinalFromBox, delay);
});
el.form.addEventListener('submit', (e) => {
  e.preventDefault();
  sendFinalFromBox(); // the box stays open after a send
});

// The type box is hidden until the Keyboard button opens it; #command keeps focus either way.
const typeBox = makeTypeBox((attrs) => {
  el.form.dataset.open = attrs['data-open'];
  el.keyboard.setAttribute('aria-expanded', attrs['aria-expanded']);
  refocus();
});
el.keyboard.addEventListener('click', () => typeBox.toggle());
// Typed characters open a hidden box so people see what they type. A first keystroke waits briefly:
// Wispr Flow dictation arrives as a burst (or one paste) and should stay on the voice line only.
let revealTimer = 0;
function revealIfTyping(kind) {
  const next = typeBox.afterInput(kind);
  if (next === 'reveal-soon' && !revealTimer) revealTimer = setTimeout(() => { revealTimer = 0; typeBox.show(); }, TYPE_REVEAL_MS);
  if (next === 'cancel-reveal') {
    clearTimeout(revealTimer);
    revealTimer = 0;
  }
}
document.addEventListener('keydown', (e) => {
  if (typeBox.closesOnKey(e.key, el.command.value)) typeBox.hide();
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
// Full screen for the demo. The text box gets focus back either way, so Wispr Flow keeps working;
// if the browser refuses, the tooltip points at Chrome's own full screen.
el.fullscreen.addEventListener('click', () => {
  const p = document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  p?.catch(() => showError('Full screen was blocked: use View → Enter Full Screen in Chrome'));
});
document.addEventListener('fullscreenchange', () => {
  document.documentElement.classList.toggle('fs', Boolean(document.fullscreenElement)); // full-screen layout (style.css)
  const label = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
  el.fullscreen.setAttribute('aria-label', label); // icon-only button: keep the icon, change the label
  el.fullscreen.title = document.fullscreenElement ? 'Exit full screen (Esc)' : 'Full screen';
  refocus(); // so Ctrl -> Wispr Flow types into the box at once, in full screen and after it
  setTimeout(refocus, 400); // again once the full-screen animation has settled
});
// Back to the starting scene (the scene reset is undoable; the picture restarts cleanly).
el.reset.addEventListener('click', () => send(tapMsg('reset')));
// Keyboard: Option+R (Alt+R). The key never types into the text box.
document.addEventListener('keydown', (e) => {
  if (!isResetShortcut(e)) return;
  e.preventDefault();
  if (!e.repeat && !el.reset.disabled) send(tapMsg('reset'));
});
// ---------- live mic ----------
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
// Spoken commands reach the server in press order; nothing is held while disconnected.
const isControlOpen = () => control?.readyState === WebSocket.OPEN;
const ordered = makeOrderedFinals(send, isControlOpen);
let rec = null; // the current recognizer; an older one may still be ending
let listening = null; // 'ptt' while push-to-talk is held, else null
let startWhenEnded = false; // only if the browser refused a second recognizer while the old one ended

function startListening(mode) {
  if (!Recognition || listening) return;
  listening = mode;
  begin(); // at once, even if the previous press is still ending: words said now are recorded
  renderMic(); // after begin(): the level meter opens its own mic stream only once recognition has started
  showPartial(''); // bars and a caret until the first words
}

// Each recognizer has its own transcriber, so a quick re-press records the new words at once while
// the previous recognizer still sends its own sentence when it ends; the two never mix.
function begin() {
  const r = new Recognition();
  const slot = ordered.open();
  // Words unchanged for SPEECH_STABLE_MS are sent as the final at once, not after Chrome's own pause.
  const endpoint = makeSpeechEndpointer(slot.send, { isOpen: isControlOpen });
  const t = makeTranscriber(endpoint.send);
  const previous = rec;
  rec = r;
  r.continuous = true;
  r.interimResults = true;
  r.lang = 'en-US';
  t.start({ hold: listening === 'ptt' }); // push-to-talk: one command per press
  r.onresult = (e) => t.handle(Array.from(e.results, (x) => ({ transcript: x[0].transcript, isFinal: x.isFinal })));
  r.onerror = (e) => {
    const text = micErrorText(e.error);
    if (!text || rec !== r) return;
    showError(text);
    listening = null; // a real failure: do not restart into it
    startWhenEnded = false;
  };
  r.onend = () => {
    t.flush(); // this recognizer's leftover words become one final, once (none if already sent early)
    endpoint.close();
    slot.close(); // ...sent only after every earlier press has sent its own
    if (rec !== r) return; // an older recognizer: the page's state belongs to the newer one
    rec = null;
    if (startWhenEnded) {
      startWhenEnded = false;
      if (listening) return begin(); // the fallback: start the press the browser refused earlier
    }
    listening = null;
    renderMic();
    endUnsent(); // the press ended with no words sent
  };
  try {
    r.start();
  } catch {
    // The browser refuses a second recognizer while the previous one is still ending: start this
    // press as soon as that one has flushed.
    endpoint.close();
    slot.close(); // this attempt recorded nothing
    rec = previous;
    startWhenEnded = true;
  }
}

function stopListening(mode) {
  if (listening !== mode) return;
  listening = null; // so onend does not restart
  startWhenEnded = false;
  renderMic();
  rec?.stop(); // onend flushes the last words
}

function renderMic() {
  el.ptt.setAttribute('aria-pressed', String(listening === 'ptt'));
  if (listening === 'ptt') startLevel();
  else stopLevel();
}

// Mic level for the voice bars and the ring around the mic, only while push-to-talk is held: a
// separate, short-lived stream next to Chrome's speech recognition, read by an AnalyserNode once per
// animation frame. It never touches recognition; if it fails, the bars keep their wave.
const level = { on: false, token: 0, audio: null, stream: null, source: null, analyser: null, data: null, raf: 0 };
const BANDS = [[2, 4], [4, 7], [7, 11], [11, 16], [16, 24]]; // FFT bins, about 375 Hz to 4.5 kHz at 48 kHz

function startLevel() {
  if (level.on || reducedMotion.matches || !navigator.mediaDevices?.getUserMedia) return;
  level.on = true;
  const token = ++level.token;
  try {
    level.audio ||= new AudioContext();
    level.audio.resume();
  } catch {
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    if (token !== level.token) return stream.getTracks().forEach((t) => t.stop()); // released meanwhile
    level.stream = stream;
    level.source = level.audio.createMediaStreamSource(stream);
    level.analyser = level.audio.createAnalyser();
    level.analyser.fftSize = 256;
    level.analyser.smoothingTimeConstant = 0.5;
    level.source.connect(level.analyser);
    level.data = new Uint8Array(level.analyser.frequencyBinCount);
    bars.classList.remove('wave');
    level.raf = requestAnimationFrame(drawLevel);
  }).catch(() => {}); // no level (refused, no device): the bars keep their wave
}

reducedMotion.addEventListener('change', () => reducedMotion.matches && stopLevel()); // turned on mid-press

function drawLevel() {
  level.analyser.getByteFrequencyData(level.data);
  let total = 0;
  BANDS.forEach(([from, to], i) => {
    let sum = 0;
    for (let k = from; k < to; k++) sum += level.data[k];
    const v = Math.min(1, (sum / (to - from) / 255) * 1.6);
    total += v;
    bars.children[i].style.setProperty('--h', (0.3 + 0.7 * v).toFixed(3));
  });
  el.pttRing.style.setProperty('--ring', (1 + (0.3 * total) / BANDS.length).toFixed(3));
  level.raf = requestAnimationFrame(drawLevel);
}

function stopLevel() {
  if (!level.on) return;
  level.on = false;
  level.token++;
  cancelAnimationFrame(level.raf);
  level.source?.disconnect();
  level.stream?.getTracks().forEach((t) => t.stop());
  level.stream = level.source = level.analyser = level.data = null;
  for (const b of bars.children) b.style.removeProperty('--h');
  el.pttRing.style.removeProperty('--ring');
  bars.classList.add('wave'); // a press still finishing shows the wave until its final lands
}

const micUnavailable = !Recognition
  ? 'This browser has no speech recognition; use Chrome'
  : !window.isSecureContext
    ? 'The mic needs HTTPS or localhost; open the page through an HTTPS tunnel or an SSH port-forward'
    : null;
const keyLabel = { Backquote: '`' }[PUSH_TO_TALK_CODE] ?? PUSH_TO_TALK_CODE;

if (micUnavailable) {
  el.ptt.disabled = true;
  el.ptt.title = micUnavailable;
} else {
  el.ptt.title = `Hold to talk (${keyLabel}) · Wispr: Ctrl`;
  el.ptt.setAttribute('aria-label', el.ptt.title);
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
showDelay(null);
showSettled();
el.timing.textContent = 'voice → video: measured after your first command';
connectControl();
connectFrames();
el.command.focus();
