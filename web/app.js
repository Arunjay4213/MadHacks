// DOM wiring. Logic lives in lib.js. Element ids are listed in PLAN.md.
import {
  PROTOCOL, finalizeDelay, partialMsg, finalMsg, tapMsg, parseFrame, parseServerMessage,
  splitKeyword, backoffMs, makeFrameGate, makeLatencyTracker, makeTranscriber,
  changesKey, commandKey, makeInputClassifier, wordsLabels, makeHeardTracker, PUSH_TO_TALK_CODE, isPushToTalk, micErrorText,
  wsBaseUrl, fitSize, videoNotice, formatStamps, isResetShortcut, makeOrderedFinals,
} from './lib.js';

const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), connection: $('connection'), banner: $('mode-banner'), words: $('words'),
  changes: $('changes'), form: $('command-form'), command: $('command'), mic: $('mic'),
  ptt: $('ptt'), undo: $('undo'), latency: $('latency'), timing: $('timing'), error: $('error'),
  notice: $('video-notice'), reset: $('reset'), frame: $('frame'), fullscreen: $('fullscreen'),
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

// Status pill: Live when connected; Reconnecting… while retrying (Connecting… before the first
// connection); Offline once several retries in a row have failed. The notice over the video says more.
const OFFLINE_AFTER = 5; // failed attempts in a row (about 8 s with the backoff)
function setConnected(open) {
  const offline = !open && attempt >= OFFLINE_AFTER;
  el.connection.dataset.state = open ? 'open' : offline ? 'offline' : 'connecting';
  el.connection.textContent = open ? 'Live' : offline ? 'Offline' : link.everConnected ? 'Reconnecting…' : 'Connecting…';
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
    if (status === 'pending') renderLocalWords(m.heard, 'partial'); // still being said or typed
    else renderWords({ ...m, confirmed: status === 'confirmed' });
  }
  latency.onServer(m);
}

// The caption: line 1 is the sentence (keyword highlighted), line 2 says what happened to it. It fades
// out a few seconds after the last update so it never sits over the picture for long.
const CAPTION_MS = 4000;
let captionTimer = 0;
function showCaption(sentenceNodes, labels, { unmatched = false } = {}) {
  const text = document.createElement('p');
  text.className = 'caption-text';
  text.append(...sentenceNodes);
  const status = document.createElement('p');
  status.className = 'caption-status';
  // Same meaning as wordsLabels (lib.js), worded for the caption.
  if (labels.label !== 'heard') [status.dataset.state, status.textContent] = ['partial', 'Listening…'];
  else if (unmatched) [status.dataset.state, status.textContent] = ['unmatched', 'Not understood'];
  else if (labels.understood) [status.dataset.state, status.textContent] = ['understood', 'Understood · applying now'];
  el.words.replaceChildren(text, ...(status.textContent ? [status] : []));
  el.words.classList.add('visible');
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => el.words.classList.remove('visible'), CAPTION_MS);
}

function renderLocalWords(text, kind) {
  showCaption([document.createTextNode(text)], wordsLabels({ local: true }));
  el.words.dataset.unmatched = 'false';
  el.words.dataset.kind = kind;
}

function renderWords({ heard, keyword, unmatched, kind, confirmed }) {
  const sentence = splitKeyword(heard, keyword).map((p) => {
    if (!p.mark) return document.createTextNode(p.text);
    const mark = document.createElement('mark');
    mark.textContent = p.text;
    return mark;
  });
  showCaption(sentence, wordsLabels({ keyword, unmatched, confirmed }), { unmatched });
  el.words.dataset.unmatched = String(Boolean(unmatched));
  el.words.dataset.kind = kind || 'server';
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

function makeChip(c) {
  const li = document.createElement('li');
  li.className = 'chip';
  li.dataset.id = c.id;
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

// Delay pill: command -> first new frame on screen, in seconds; amber over 1.5 s, red over 3 s. The
// server's own stage stamps show on hover; the exact definition is in the tooltip.
function showDelay(ms) {
  el.latency.textContent = ms === null ? '— delay' : `${(ms / 1000).toFixed(1)} s delay`;
  el.latency.dataset.level = ms === null ? '' : ms > 3000 ? 'bad' : ms > 1500 ? 'slow' : 'ok';
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
  const text = el.command.value.trim();
  if (!text) return heardTracker.clearPartials(); // draft deleted: it is no longer pending
  send(partialMsg(text));
  // Sent after a pause, unless it still looks unfinished ("a car,"): then only Enter sends it.
  const delay = finalizeDelay(kind, text);
  if (delay !== null) idleTimer = setTimeout(sendFinalFromBox, delay);
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
  el.fullscreen.title = document.fullscreenElement ? 'Exit full screen (Esc)' : 'Full screen for the demo (Esc leaves)';
  refocus();
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
const ordered = makeOrderedFinals(send, () => control?.readyState === WebSocket.OPEN);
let rec = null; // the current recognizer; an older one may still be ending
let listening = null; // what the user asked for: 'toggle' | 'ptt' | null
let startedAt = 0;
let startWhenEnded = false; // only if the browser refused a second recognizer while the old one ended

function startListening(mode) {
  if (!Recognition || listening) return;
  listening = mode;
  renderMic();
  begin(); // at once, even if the previous press is still ending: words said now are recorded
}

// Each recognizer has its own transcriber, so a quick re-press records the new words at once while
// the previous recognizer still sends its own sentence when it ends; the two never mix.
function begin() {
  const r = new Recognition();
  const slot = ordered.open();
  const t = makeTranscriber(slot.send);
  const previous = rec;
  rec = r;
  r.continuous = true;
  r.interimResults = true;
  r.lang = 'en-US';
  t.start({ hold: listening === 'ptt' }); // push-to-talk: one command per press
  startedAt = performance.now();
  r.onresult = (e) => t.handle(Array.from(e.results, (x) => ({ transcript: x[0].transcript, isFinal: x.isFinal })));
  r.onerror = (e) => {
    const text = micErrorText(e.error);
    if (!text || rec !== r) return;
    showError(text);
    listening = null; // a real failure: do not restart into it
    startWhenEnded = false;
  };
  r.onend = () => {
    t.flush(); // this recognizer's leftover words become one final, once
    slot.close(); // ...sent only after every earlier press has sent its own
    if (rec !== r) return; // an older recognizer: the page's state belongs to the newer one
    rec = null;
    if (startWhenEnded) {
      startWhenEnded = false;
      if (listening) return begin(); // the fallback: start the press the browser refused earlier
    }
    // Chrome ends recognition by itself after a pause; keep listening while the mic toggle is on,
    // unless it ended at once (then restarting would just spin).
    if (listening === 'toggle' && performance.now() - startedAt > 500) return begin();
    listening = null;
    renderMic();
  };
  try {
    r.start();
  } catch {
    // The browser refuses a second recognizer while the previous one is still ending: start this
    // press as soon as that one has flushed.
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
showDelay(null);
el.timing.textContent = 'voice → video: measured after your first command';
connectControl();
connectFrames();
el.command.focus();
