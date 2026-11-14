// Pure logic for the page. No DOM, so node --test can load it.

// Wire details from contracts/README.md ("Wire details"), agreed by the lead.
export const PROTOCOL = {
  controlPath: '/control', // contract 1 in, contract 4 out, JSON text
  framesPath: '/frames', // binary: uint32 BE chunk + uint16 BE frame index + JPEG bytes
  frameHeaderBytes: 6,
};

// Typed text becomes final after this pause; machine-fast input (a dictation tool simulating keys)
// after the much shorter BURST_IDLE_MS, so it is not held back by a pause meant for humans.
export const TYPING_IDLE_MS = 1500;
export const BURST_IDLE_MS = 200;

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
export function makeLatencyTracker(maxSent = 50) {
  const sent = [];
  let pending = null;
  return {
    sent(t) {
      sent.push(t);
      if (sent.length > maxSent) sent.shift();
    },
    onServer(m) {
      if (isNum(m.t_client) && isChunk(m.applies_from_chunk) && sent.includes(m.t_client)) {
        pending = { t: m.t_client, chunk: m.applies_from_chunk, stamps: m.stamps };
      }
    },
    // Returns {ms, stamps} once per correlated command, else null.
    onFrameDrawn(chunk, now = Date.now()) {
      if (!pending || chunk < pending.chunk) return null;
      const r = { ms: Math.round(now - pending.t), stamps: pending.stamps };
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
export function makeTranscriber(send) {
  let committed = new Set();
  let lastInterim = '';
  return {
    start() {
      committed = new Set();
      lastInterim = '';
    },
    handle(results) {
      let interim = '';
      results.forEach((r, i) => {
        const text = r.transcript.trim();
        if (r.isFinal) {
          if (!committed.has(i)) {
            committed.add(i);
            if (text) send(finalMsg(text));
          }
        } else {
          interim += (interim ? ' ' : '') + text;
        }
      });
      if (interim !== lastInterim) {
        lastInterim = interim;
        if (interim) send(partialMsg(interim));
      }
    },
    // Called once recognition has stopped: leftover interim text becomes one final.
    flush() {
      const text = lastInterim;
      lastInterim = '';
      if (text) send(finalMsg(text));
    },
  };
}
