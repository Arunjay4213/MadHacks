import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseFrame, parseServerMessage, splitKeyword, backoffMs, isWholeInsert,
  makeFrameGate, makeLatencyTracker, makeTranscriber, changesKey, commandKey, makeInputClassifier,
  PUSH_TO_TALK_CODE, isPushToTalk, micErrorText, looksUnfinished, finalizeDelay,
} from '../lib.js';
import { loadTestPattern } from '../fake-server.mjs';
import { wsBaseUrl, fitSize, videoNotice, formatStamps, STALL_MS, isResetShortcut } from '../lib.js';

test('reset shortcut: Option/Alt+R only; never with Ctrl (Wispr Flow) or Cmd; plain R just types', () => {
  const key = (code, mods = {}) => ({ code, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...mods });
  assert.equal(isResetShortcut(key('KeyR', { altKey: true })), true);
  assert.equal(isResetShortcut(key('KeyR')), false);
  assert.equal(isResetShortcut(key('KeyR', { altKey: true, ctrlKey: true })), false);
  assert.equal(isResetShortcut(key('KeyR', { altKey: true, metaKey: true })), false);
  assert.equal(isResetShortcut(key('KeyT', { altKey: true })), false);
});

test('videoNotice: connecting, lost, waiting, paused, or live', () => {
  const live = { controlOpen: true, framesOpen: true, lastFrameAt: 1000, now: 1100, everConnected: true };
  assert.equal(videoNotice(live), null);
  assert.match(videoNotice({ ...live, controlOpen: false, everConnected: false }), /Connecting/);
  assert.match(videoNotice({ ...live, framesOpen: false }), /Connection lost/); // either socket down
  assert.match(videoNotice({ ...live, lastFrameAt: null }), /first frame/);
  assert.match(videoNotice({ ...live, now: 1000 + STALL_MS + 1 }), /paused/);
  assert.equal(videoNotice({ ...live, now: 1000 + STALL_MS }), null);
});

test('formatStamps: pipeline order, known stamps only', () => {
  assert.equal(formatStamps({ heard: 0, shown: 1010, decided: 4, generated: 790, embedded: 61 }), 'decided 4 · embedded 61 · generated 790 · shown 1010');
  assert.equal(formatStamps({ heard: 0, decided: 3 }), 'decided 3');
  assert.equal(formatStamps(undefined), '');
});

test('wsBaseUrl: same origin by default; ?server= as host:port or a full URL (tunnels)', () => {
  const local = { protocol: 'http:', host: 'localhost:8787' };
  assert.equal(wsBaseUrl(null, local), 'ws://localhost:8787');
  assert.equal(wsBaseUrl('127.0.0.1:8000', local), 'ws://127.0.0.1:8000');
  assert.equal(wsBaseUrl('wss://engine.example.trycloudflare.com/', local), 'wss://engine.example.trycloudflare.com');
  assert.equal(wsBaseUrl('https://engine.example.trycloudflare.com', local), 'wss://engine.example.trycloudflare.com');
  assert.equal(wsBaseUrl('10.0.0.5:8000', { protocol: 'https:', host: 'x' }), 'wss://10.0.0.5:8000');
});

test('fitSize: real 672x384 frames fill the screen at their own aspect ratio, in device pixels', () => {
  // A 1880x900 CSS-pixel video area on a 2x Retina screen: height-limited.
  assert.deepEqual(fitSize(672, 384, 1880, 900, 2), { width: 3150, height: 1800 });
  // A wide area: aspect ratio kept (672/384 = 1.75).
  const s = fitSize(672, 384, 3000, 600, 1);
  assert.equal(s.height, 600);
  assert.equal(s.width, 1050);
  // The 416x240 test pattern and odd inputs.
  assert.deepEqual(fitSize(416, 240, 832, 480, 1), { width: 832, height: 480 });
  assert.deepEqual(fitSize(672, 384, 0, 0, 1), { width: 672, height: 384 }); // not laid out yet
});

test('push-to-talk: the set key alone; never with Ctrl (Wispr Flow), Cmd or Alt; never Ctrl itself', () => {
  const key = (code, mods = {}) => ({ code, ctrlKey: false, metaKey: false, altKey: false, ...mods });
  assert.ok(!/Control/.test(PUSH_TO_TALK_CODE));
  assert.equal(isPushToTalk(key(PUSH_TO_TALK_CODE)), true);
  assert.equal(isPushToTalk(key(PUSH_TO_TALK_CODE, { ctrlKey: true })), false);
  assert.equal(isPushToTalk(key(PUSH_TO_TALK_CODE, { metaKey: true })), false);
  assert.equal(isPushToTalk(key(PUSH_TO_TALK_CODE, { altKey: true })), false);
  assert.equal(isPushToTalk(key('ControlLeft', { ctrlKey: true })), false);
  assert.equal(isPushToTalk(key('KeyA')), false);
  assert.equal(isPushToTalk(key('F13'), 'F13'), true); // the setting can be changed
});

test('micErrorText explains blocked mic and network, hides routine ends', () => {
  assert.equal(micErrorText('no-speech'), null);
  assert.equal(micErrorText('aborted'), null);
  assert.match(micErrorText('not-allowed'), /blocked/);
  assert.match(micErrorText('network'), /internet/);
});

test('input classifier: paste is whole, machine-fast keys are a burst, human keys are typing', () => {
  const c = makeInputClassifier();
  assert.equal(c.classify('insertFromPaste', null, 0), 'whole');
  // Human typing: about 120 ms between keys.
  assert.deepEqual([0, 120, 240, 360, 480].map((t) => c.classify('insertText', 'x', t)), ['typing', 'typing', 'typing', 'typing', 'typing']);
  c.reset();
  // Simulated keys 5 ms apart: a burst once four have arrived.
  assert.deepEqual([1000, 1005, 1010, 1015, 1020].map((t) => c.classify('insertText', 'x', t)), ['typing', 'typing', 'typing', 'burst', 'burst']);
  // One slow key ends the burst.
  assert.equal(c.classify('insertText', 'x', 1500), 'typing');
});

test('parseFrame reads big-endian chunk, frame index and JPEG bytes; rejects short or non-buffer input', () => {
  const buf = new Uint8Array([0, 0, 1, 2, 0, 3, 0xff, 0xd8]).buffer;
  const f = parseFrame(buf);
  assert.equal(f.chunk, 258);
  assert.equal(f.frame, 3);
  assert.deepEqual([...f.jpeg], [0xff, 0xd8]);
  assert.equal(parseFrame(new Uint8Array([0, 0, 0, 1, 0, 0]).buffer), null); // header only, no JPEG
  assert.equal(parseFrame('nope'), null);
});

test('changesKey and commandKey only change when what is shown would change', () => {
  const a = [{ id: 'c1', label: 'winter', strength: 1 }];
  assert.equal(changesKey(a), changesKey([{ id: 'c1', label: 'winter', strength: 1 }]));
  assert.notEqual(changesKey(a), changesKey([{ id: 'c1', label: 'winter', strength: 0.5 }]));
  assert.notEqual(changesKey(a), changesKey([]));
  const m = { t_client: 5, heard: 'make it winter', keyword: 'winter', unmatched: false, chunk: 1 };
  assert.equal(commandKey(m), commandKey({ ...m, chunk: 2 })); // a new chunk alone is not a new command
  assert.notEqual(commandKey(m), commandKey({ ...m, t_client: 6 }));
});

test('parseServerMessage accepts contract 4 and drops malformed optional fields', () => {
  const m = parseServerMessage(JSON.stringify({
    chunk: 412, changes: [{ id: 'c1', label: 'winter', strength: 0.5 }], progress: 0.5,
    stamps: { heard: 0, decided: 4, bogus: 'x' }, t_client: 'nope', applies_from_chunk: -1, fake: true,
  }));
  assert.deepEqual(m, { chunk: 412, changes: [{ id: 'c1', label: 'winter', strength: 0.5 }], progress: 0.5, stamps: { heard: 0, decided: 4 }, fake: true });
});

test('parseServerMessage rejects garbage and malformed changes without throwing', () => {
  for (const raw of ['', 'not json', 'null', '[]', '42', '"str"',
    JSON.stringify({ changes: 'x' }), JSON.stringify({ changes: [{ id: 1, label: 'a' }] }), JSON.stringify({ changes: [null] })]) {
    assert.equal(parseServerMessage(raw), null, raw);
  }
});

test('splitKeyword marks the keyword case-insensitively', () => {
  assert.deepEqual(splitKeyword('Make it Winter now', 'winter'), [
    { text: 'Make it ', mark: false }, { text: 'Winter', mark: true }, { text: ' now', mark: false }]);
  assert.deepEqual(splitKeyword('hello', 'snow'), [{ text: 'hello', mark: false }]);
  assert.deepEqual(splitKeyword('hello', undefined), [{ text: 'hello', mark: false }]);
});

test('backoff doubles and caps at 5 s', () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 10].map(backoffMs), [250, 500, 1000, 2000, 4000, 5000, 5000]);
});

test('isWholeInsert: paste and multi-char inserts are whole sentences, keystrokes are not', () => {
  assert.equal(isWholeInsert('insertFromPaste', null), true);
  assert.equal(isWholeInsert('insertReplacementText', null), true);
  assert.equal(isWholeInsert('insertText', 'make it winter'), true);
  assert.equal(isWholeInsert('insertText', 'm'), false);
  assert.equal(isWholeInsert('deleteContentBackward', null), false);
});

test('frame gate never draws an older frame after a newer one (out-of-order decode)', () => {
  const g = makeFrameGate();
  const a = g.next(), b = g.next(), c = g.next();
  assert.equal(g.tryDraw(b), true);
  assert.equal(g.tryDraw(a), false); // decoded late
  assert.equal(g.tryDraw(c), true);
  assert.equal(g.tryDraw(c), false);
});

test('latency is unavailable without correlation, and only for t_client this page sent', () => {
  const l = makeLatencyTracker();
  assert.equal(l.onFrameDrawn(5, 2000), null);
  l.onServer({ t_client: 1000, applies_from_chunk: 5 }); // never sent: ignored
  assert.equal(l.onFrameDrawn(9, 2000), null);
  l.sent(1000);
  l.onServer({ t_client: 1000 }); // no chunk: ignored
  assert.equal(l.onFrameDrawn(9, 2000), null);
  l.onServer({ t_client: 1000, applies_from_chunk: 5, stamps: { decided: 3 } });
  assert.equal(l.onFrameDrawn(4, 1500), null); // older chunk
  assert.deepEqual(l.onFrameDrawn(5, 1810), { ms: 810, stamps: { decided: 3 } });
  assert.equal(l.onFrameDrawn(6, 1900), null); // reported once
  l.onServer({ t_client: 1000, applies_from_chunk: 7 });
  l.reset(); // disconnect
  assert.equal(l.onFrameDrawn(8, 2000), null);
});

test('latency: the per-chunk echo of the same t_client is measured once, never again', () => {
  const l = makeLatencyTracker();
  l.sent(1000);
  // The server repeats the most recent command in every per-chunk message.
  l.onServer({ t_client: 1000, applies_from_chunk: 5 });
  l.onServer({ t_client: 1000, applies_from_chunk: 6 }); // a repeat must not move the target chunk
  assert.deepEqual(l.onFrameDrawn(5, 1400), { ms: 400, stamps: undefined });
  // 35 s later the same command is still being echoed: no new number (this was the 35578 ms bug).
  for (let chunk = 6; chunk < 10; chunk++) l.onServer({ t_client: 1000, applies_from_chunk: 5 });
  assert.equal(l.onFrameDrawn(140, 36578), null);
  // The next command is measured normally.
  l.sent(40000);
  l.onServer({ t_client: 40000, applies_from_chunk: 150 });
  assert.equal(l.onFrameDrawn(149, 40100), null);
  assert.deepEqual(l.onFrameDrawn(150, 40300), { ms: 300, stamps: undefined });
});

test('latency: an unmatched command is never measured, even with a chunk number attached', () => {
  const l = makeLatencyTracker();
  l.sent(5000);
  // An engine that echoes the unmatched command's t_client with the previous command's chunk.
  l.onServer({ t_client: 5000, applies_from_chunk: 3, unmatched: true });
  assert.equal(l.onFrameDrawn(9, 5100), null);
});

test('fake server test pattern: real JPEG frames bigger than one pixel', () => {
  const frames = loadTestPattern();
  assert.ok(frames.length >= 16);
  for (const f of frames) {
    assert.deepEqual([...f.subarray(0, 2)], [0xff, 0xd8]); // JPEG start-of-image
    // Find the frame header (SOF0, 0xFFC0) and read the height and width stored in it.
    const i = f.indexOf(Buffer.from([0xff, 0xc0]));
    assert.ok(i > 0);
    assert.ok(f.readUInt16BE(i + 5) > 100 && f.readUInt16BE(i + 7) > 100);
  }
});

test('looksUnfinished / finalizeDelay: a sentence still being typed is never sent on a pause', () => {
  for (const t of ['a car,', 'make it night and', 'add a', 'a car racing on', 'make it', 'add a red car -']) {
    assert.equal(looksUnfinished(t), true, t);
    assert.equal(finalizeDelay('typing', t), null, t);
    assert.equal(finalizeDelay('burst', t), null, t);
  }
  for (const t of ['a car, car racing on a road', 'make it winter', 'go back', 'add a river.', 'undo']) {
    assert.equal(looksUnfinished(t), false, t);
  }
  assert.equal(finalizeDelay('typing', 'make it winter'), 2500);
  assert.equal(finalizeDelay('burst', 'make it winter'), 600);
});

test('transcriber hold mode (push-to-talk): Chrome splitting at a pause still sends ONE final on release', () => {
  const out = [];
  const t = makeTranscriber((m) => out.push([m.type, m.text]));
  t.start({ hold: true });
  t.handle([R('make it night')]);
  t.handle([R('make it night', true)]); // Chrome ends a segment at the pause
  t.handle([R('make it night', true), R('and add a car')]);
  t.handle([R('make it night', true), R('and add a car racing on a road', true)]);
  t.flush(); // key released
  assert.deepEqual(out.filter(([type]) => type === 'final'), [['final', 'make it night and add a car racing on a road']]);
  assert.deepEqual(out.at(-2), ['partial', 'make it night and add a car racing on a road']); // words line showed it all
});

const R = (transcript, isFinal = false) => ({ transcript, isFinal });

test('transcriber: interim revisions send partials only on change; each final once', () => {
  const out = [];
  const t = makeTranscriber((m) => out.push([m.type, m.text]));
  t.start();
  t.handle([R('make it')]);
  t.handle([R('make it')]); // no change: nothing
  t.handle([R('make it win')]);
  t.handle([R('make it winter', true)]);
  t.handle([R('make it winter', true)]); // same final again: not resent
  t.handle([R('make it winter', true), R('also')]);
  t.handle([R('make it winter', true), R('also add a river', true)]);
  t.flush(); // nothing left
  assert.deepEqual(out, [
    ['partial', 'make it'], ['partial', 'make it win'], ['final', 'make it winter'],
    ['partial', 'also'], ['final', 'also add a river']]);
});

test('transcriber: stopping flushes leftover interim as one final, once', () => {
  const out = [];
  const t = makeTranscriber((m) => out.push([m.type, m.text]));
  t.start();
  t.handle([R('go ba')]);
  t.flush();
  t.flush();
  assert.deepEqual(out, [['partial', 'go ba'], ['final', 'go ba']]);
});

test('transcriber: a new session restarts result indices', () => {
  const out = [];
  const t = makeTranscriber((m) => out.push(m.text));
  t.start();
  t.handle([R('undo', true)]);
  t.start();
  t.handle([R('undo', true)]); // index 0 again in the new session: a new command
  assert.deepEqual(out, ['undo', 'undo']);
});
