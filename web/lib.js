// Pure logic for the page. No DOM, so node --test can load it.

// Wire details from contracts/README.md ("Wire details"), agreed by the lead.
export const PROTOCOL = {
  controlPath: '/control', // contract 1 in, contract 4 out, JSON text
  framesPath: '/frames', // binary: uint32 BE chunk + uint16 BE frame index + JPEG bytes
  frameHeaderBytes: 6,
};

// Typed text becomes final after this pause; machine-fast input (a dictation tool simulating keys)
// after the much shorter BURST_IDLE_MS, so it is not held back by a pause meant for humans.
export const TYPING_IDLE_MS = 2500;
export const BURST_IDLE_MS = 600;

// Typed text that ends like this is a sentence still being written ("a car," / "make it night and"),
// so it is never sent on a pause, only on Enter: sending it would split one command into two.
// Words that can END a real command ("start over", "undo it", "make it night slowly", "more snow" / "snow
// more") are deliberately not here, or that command would never send on a pause (Greptile on #82).
const CONNECTORS = new Set(['a', 'an', 'the', 'and', 'or', 'but', 'then', 'also', 'plus', 'with', 'on', 'in', 'into',
  'onto', 'to', 'of', 'at', 'by', 'for', 'from', 'through', 'under', 'near', 'along', 'across', 'make', 'add',
  'turn', 'remove', 'is', 'some', 'very', 'really']);
export function looksUnfinished(text) {
  const t = text.trim().toLowerCase();
  if (!t) return false;
  if (/[,;:\-–—]$/.test(t)) return true;
  const words = t.split(/\s+/).map((w) => w.replace(/[^a-z']/g, ''));
  const last = words.at(-1);
  // "it" ends "undo it" / "remove it", but "make it" / "turn it" are still waiting for what it becomes.
  if (last === 'it') return ['make', 'turn', 'put', 'set'].includes(words.at(-2));
  return CONNECTORS.has(last);
}

// How long typed text waits before it is sent as a command; null means wait for Enter.
export const finalizeDelay = (kind, text) =>
  looksUnfinished(text) ? null : kind === 'burst' ? BURST_IDLE_MS : TYPING_IDLE_MS;

// Push-to-talk for the live mic: hold this key (a KeyboardEvent.code). Change it here.
// Never Ctrl or a Ctrl combination: Ctrl is Wispr Flow's hotkey on the demo laptop.
export const PUSH_TO_TALK_CODE = 'Backquote';

// True for a press of the push-to-talk key with no modifier held, so Ctrl (Wispr Flow),
// Cmd and Alt shortcuts never start the mic.
export const isPushToTalk = (e, code = PUSH_TO_TALK_CODE) =>
  e.code === code && !e.ctrlKey && !e.metaKey && !e.altKey;

// Reset shortcut: Option+R on a Mac (Alt+R elsewhere). Not Ctrl (Wispr Flow), and with Option held
// the key does not type into the always-focused text box (the page blocks the ® it would insert).
export const RESET_SHORTCUT_CODE = 'KeyR';
export const isResetShortcut = (e, code = RESET_SHORTCUT_CODE) =>
  e.code === code && e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey;

// Plain-language text for Chrome speech recognition errors; null for ones not worth showing.
export function micErrorText(error) {
  if (error === 'no-speech' || error === 'aborted') return null;
  if (error === 'not-allowed' || error === 'service-not-allowed') return 'Microphone blocked: allow it for this site in Chrome (lock icon in the address bar).';
  if (error === 'network') return 'Speech recognition needs the internet (Chrome sends audio to Google). Use the text box or Wispr Flow.';
  if (error === 'audio-capture') return 'No microphone found.';
  return `Mic: ${error}`;
}

export const partialMsg = (text, t = Date.now()) => ({ type: 'partial', text, t_client: t });
export const finalMsg = (text, t = Date.now()) => ({ type: 'final', text, t_client: t });
export const tapMsg = (action, id) => (id === undefined ? { type: 'tap', action } : { type: 'tap', action, id });

export function parseFrame(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength <= PROTOCOL.frameHeaderBytes) return null;
  const view = new DataView(buf);
  return {
    chunk: view.getUint32(0, false),
    frame: view.getUint16(4, false),
    jpeg: new Uint8Array(buf, PROTOCOL.frameHeaderBytes),
  };
}

// The server sends state once per chunk (4 times a second). Rebuilding the chips on every message
// would swap the buttons out from under a click, so the page re-renders only when this key changes.
export const changesKey = (changes) => JSON.stringify(changes.map((c) => [c.id, c.label, c.strength ?? null]));
// Same idea for the words line: a new command has a new t_client or new heard text.
export const commandKey = (m) => JSON.stringify([m.t_client ?? null, m.heard ?? null, m.keyword ?? null, m.unmatched ?? null]);

const isStr = (v) => typeof v === 'string';
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isChunk = (v) => Number.isInteger(v) && v >= 0;

// Contract 4 (plus provisional fields). Returns a clean object, or null if the payload is unusable.
// Optional fields that are malformed are dropped; a malformed `changes` rejects the whole message,
// because chips must never show a guessed state.
export function parseServerMessage(raw) {
  let m;
  try {
    m = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!m || typeof m !== 'object' || Array.isArray(m)) return null;
  const out = {};
  if ('changes' in m) {
    if (!Array.isArray(m.changes)) return null;
    const changes = [];
    for (const c of m.changes) {
      if (!c || !isStr(c.id) || !isStr(c.label)) return null;
      changes.push({ id: c.id, label: c.label, ...(isNum(c.strength) ? { strength: c.strength } : {}) });
    }
    out.changes = changes;
  }
  if (isChunk(m.chunk)) out.chunk = m.chunk;
  if (isChunk(m.applies_from_chunk)) out.applies_from_chunk = m.applies_from_chunk;
  if (isNum(m.t_client)) out.t_client = m.t_client;
  if (isNum(m.progress)) out.progress = m.progress;
  if (isStr(m.heard)) out.heard = m.heard;
  if (isStr(m.keyword)) out.keyword = m.keyword;
  if (m.kind === 'partial' || m.kind === 'final') out.kind = m.kind;
  if (typeof m.unmatched === 'boolean') out.unmatched = m.unmatched;
  if (typeof m.fake === 'boolean') out.fake = m.fake;
  if (isStr(m.error)) out.error = m.error;
  if (m.stamps && typeof m.stamps === 'object' && !Array.isArray(m.stamps)) {
    const stamps = Object.fromEntries(Object.entries(m.stamps).filter(([, v]) => isNum(v)));
    if (Object.keys(stamps).length) out.stamps = stamps;
  }
  return out;
}

// Splits heard text around the first case-insensitive keyword match, for highlighting.
export function splitKeyword(text, keyword) {
  const i = keyword ? text.toLowerCase().indexOf(keyword.toLowerCase()) : -1;
  if (i < 0) return [{ text, mark: false }];
  return [
    { text: text.slice(0, i), mark: false },
    { text: text.slice(i, i + keyword.length), mark: true },
    { text: text.slice(i + keyword.length), mark: false },
  ].filter((p) => p.text);
}

export const backoffMs = (attempt) => Math.min(5000, 250 * 2 ** attempt);

// What the words line says around the sentence: a label before it ("hearing…" while you type or speak,
// "heard" once the server has it) and, for the server's version, what it understood.
// `confirmed` is true only for a final this page sent: the server also reports partial text with
// unmatched false, so a no-keyword "understood" is claimed only after a confirmed final.
export function wordsLabels({ local, keyword, unmatched, confirmed = false }) {
  if (local) return { label: 'hearing…', understood: '' };
  if (unmatched) return { label: 'heard', understood: '→ not understood' };
  if (keyword) return { label: 'heard', understood: `→ understood: ${keyword}` };
  // No keyword when the AI reader (or another fallback) understood it: say so, but only for a confirmed final.
  return { label: 'heard', understood: confirmed ? '→ understood' : '' };
}

// Tells whether the server's "heard" is this page's own partial (still being said or typed), this
// page's own final (confirmed), or something else (another viewer's command, or the starting state).
// A partial counts as still being said or typed only for PENDING_MS after it was sent, so an abandoned
// draft never hides another viewer's identical command.
export const PENDING_MS = 5000;
export function makeHeardTracker(max = 50) {
  let partials = new Map(); // text -> time sent
  const finals = [];
  return {
    sentPartial(text, now = Date.now()) {
      partials.set(text, now);
    },
    sentFinal(text, t) {
      partials = new Map(); // the sentence is done; its partials no longer count as pending
      finals.push({ text, t });
      if (finals.length > max) finals.shift();
    },
    clearPartials() {
      partials = new Map(); // the draft was deleted or abandoned
    },
    classify(heard, tClient, now = Date.now()) {
      if (finals.some((f) => f.t === tClient && f.text === heard)) return 'confirmed';
      const sentAt = partials.get(heard);
      if (sentAt !== undefined && now - sentAt < PENDING_MS) return 'pending';
      return 'other';
    },
  };
}

// How long without a frame, while connected, before the page says the video has paused.
export const STALL_MS = 2000;

// The notice shown over the video, or null when the video is live.
export function videoNotice({ controlOpen, framesOpen, lastFrameAt, now, everConnected }) {
  if (!controlOpen || !framesOpen) return everConnected ? 'Connection lost, reconnecting…' : 'Connecting to the video server…';
  if (lastFrameAt === null) return 'Waiting for the first frame…';
  if (now - lastFrameAt > STALL_MS) return 'Video paused: no frames from the server';
  return null;
}

// Server stamps (ms after the server heard the command) as one short line, in pipeline order.
export function formatStamps(stamps) {
  if (!stamps) return '';
  return ['decided', 'embedded', 'generated', 'shown'].filter((k) => isNum(stamps[k])).map((k) => `${k} ${stamps[k]}`).join(' · ');
}

// Websocket base URL from the page location and an optional `?server=` value: either `host:port`
// (ws, or wss on an https page) or a full `ws://` / `wss://` / `http(s)://` URL, e.g. an HTTPS tunnel.
export function wsBaseUrl(server, location) {
  if (!server) return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
  if (/^wss?:\/\//.test(server)) return server.replace(/\/+$/, '');
  if (/^https?:\/\//.test(server)) return server.replace(/^http/, 'ws').replace(/\/+$/, '');
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${server}`;
}

// Canvas size in device pixels that fits a frame of fw x fh inside a box of boxW x boxH CSS pixels,
// keeping the frame's aspect ratio. The page draws each frame scaled to this size with high-quality
// smoothing, so a small frame (672x384) is upscaled smoothly for a big screen.
export function fitSize(fw, fh, boxW, boxH, dpr = 1) {
  if (!(fw > 0 && fh > 0 && boxW > 0 && boxH > 0)) return { width: fw || 1, height: fh || 1 };
  const scale = Math.min(boxW / fw, boxH / fh) * dpr;
  return { width: Math.max(1, Math.round(fw * scale)), height: Math.max(1, Math.round(fh * scale)) };
}

// Text box: a paste or Wispr Flow insert is a whole sentence, so it is final at once.
export function isWholeInsert(inputType, data) {
  if (['insertFromPaste', 'insertFromDrop', 'insertReplacementText', 'insertFromYank'].includes(inputType)) return true;
  return inputType === 'insertText' && isStr(data) && data.length > 1;
}

// Tells human typing from a burst of simulated keystrokes: `burstCount` inputs in a row, each within
// `burstGapMs` of the last, are faster than anyone types. Returns 'whole' | 'burst' | 'typing'.
export function makeInputClassifier({ burstGapMs = 30, burstCount = 4 } = {}) {
  let times = [];
  return {
    classify(inputType, data, now) {
      if (isWholeInsert(inputType, data)) {
        times = [];
        return 'whole';
      }
      times.push(now);
      if (times.length > burstCount) times.shift();
      const burst = times.length === burstCount && times.every((t, i) => i === 0 || t - times[i - 1] <= burstGapMs);
      return burst ? 'burst' : 'typing';
    },
    reset() {
      times = [];
    },
  };
}

// Async JPEG decode can finish out of order; only draw a frame newer than the last drawn one.
export function makeFrameGate() {
  let received = 0;
  let drawn = 0;
  return {
    next: () => ++received,
    tryDraw(seq) {
      if (seq <= drawn) return false;
      drawn = seq;
      return true;
    },
  };
}

// Delay = page-clock time a frame tagged with chunk >= applies_from_chunk is drawn, minus the
// command's t_client. Only for t_client values this page sent, so an echo cannot invent a number.
// The server repeats the most recent command's t_client in every per-chunk message, so each
// command is measured once: a t_client already pending or already reported is ignored.
export function makeLatencyTracker(maxSent = 50) {
  const sent = [];
  let pending = null;
  return {
    sent(t) {
      sent.push(t);
      if (sent.length > maxSent) sent.shift();
    },
    onServer(m) {
      if (!isNum(m.t_client) || !isChunk(m.applies_from_chunk) || !sent.includes(m.t_client)) return;
      if (m.unmatched === true) return; // nothing changed, so there is no delay to measure
      if (pending?.t === m.t_client) {
        pending.stamps = m.stamps; // same command: keep its target chunk, take the newest stamps
        return;
      }
      pending = { t: m.t_client, chunk: m.applies_from_chunk, stamps: m.stamps };
    },
    // Returns {ms, stamps} once per correlated command, else null.
    onFrameDrawn(chunk, now = Date.now()) {
      if (!pending || chunk < pending.chunk) return null;
      const r = { ms: Math.round(now - pending.t), stamps: pending.stamps };
      sent.splice(sent.indexOf(pending.t), 1); // reported: later echoes of it are ignored
      pending = null;
      return r;
    },
    reset() {
      sent.length = 0;
      pending = null;
    },
  };
}

// Speech recognition results -> contract-1 messages, with no duplicate finals.
// `results` is [{transcript, isFinal}] for the current recognition session.
// Keeps spoken commands in the order the presses happened. Each press opens a slot; partials go out at
// once (the live words line), but a press's finals wait until every earlier press has closed, so a quick
// second press ("undo") that ends first can never reach the server before the command it undoes.
export function makeOrderedFinals(send) {
  const slots = [];
  const drain = () => {
    while (slots.length && slots[0].closed) {
      for (const m of slots.shift().finals) send(m);
    }
  };
  return {
    open() {
      const slot = { finals: [], closed: false };
      slots.push(slot);
      return {
        send(msg) {
          if (msg.type !== 'final') return send(msg);
          if (slots[0] === slot) return send(msg); // nothing earlier is waiting
          slot.finals.push(msg);
          return true;
        },
        close() {
          slot.closed = true;
          drain();
        },
      };
    },
  };
}

// With { hold: true } (push-to-talk) Chrome's final segments are kept, not sent: Chrome splits speech
// at pauses, and one press of the key is one command, so flush() sends the whole utterance once.
// Partials then show the utterance so far.
export function makeTranscriber(send) {
  let committed = new Set();
  let lastInterim = '';
  let hold = false;
  let held = [];
  return {
    start({ hold: h = false } = {}) {
      committed = new Set();
      lastInterim = '';
      hold = h;
      held = [];
    },
    handle(results) {
      let interim = '';
      results.forEach((r, i) => {
        const text = r.transcript.trim();
        if (r.isFinal) {
          if (!committed.has(i)) {
            committed.add(i);
            if (text && hold) held.push(text);
            else if (text) send(finalMsg(text));
          }
        } else {
          interim += (interim ? ' ' : '') + text;
        }
      });
      const shown = hold ? [...held, interim].filter(Boolean).join(' ') : interim;
      if (shown !== lastInterim) {
        lastInterim = shown;
        if (shown) send(partialMsg(shown));
      }
    },
    // Called once recognition has stopped: leftover text becomes one final, once. In hold mode the
    // last partial already is the whole utterance (held finals plus the current interim).
    flush() {
      const text = lastInterim;
      lastInterim = '';
      held = [];
      if (text) send(finalMsg(text));
    },
  };
}
