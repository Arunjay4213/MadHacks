import { describe, expect, it } from 'vitest';
import { TranscriptGate, makeInputClassifier, isPushToTalk, isResetShortcut, finalizeDelay } from './input';
import { parseEngineState, engineBaseUrl, videoNotice, makeLatencyTracker, makeHeardTracker } from './engine';

describe('web input parity', () => {
  it('flushes leftover speech once, without replaying a committed final', () => {
    const gate = new TranscriptGate();
    gate.accept(0, 'make it winter', false);
    expect(gate.flush()).toBe('make it winter');
    expect(gate.flush()).toBeNull();
    gate.accept(0, 'add a river', false);
    gate.accept(0, 'add a river', true);
    expect(gate.flush()).toBeNull();
  });
  it('keeps a held utterance together across Chrome final segments', () => {
    const gate = new TranscriptGate();
    gate.reset(true);
    expect(gate.accept(0, 'make it night', true)).toBe('partial');
    expect(gate.accept(1, 'and add a car', false)).toBe('partial');
    expect(gate.interim()).toBe('make it night and add a car');
    gate.accept(1, 'and add a car racing on a road', true);
    expect(gate.flush()).toBe('make it night and add a car racing on a road');
    expect(gate.flush()).toBeNull();
  });
  it('holds unfinished typed sentences, while completed reset and pacing phrases still send', () => {
    for (const text of ['a car,', 'make it night and', 'add a', 'a car racing on']) {
      expect(finalizeDelay('typing', text)).toBeNull();
      expect(finalizeDelay('burst', text)).toBeNull();
    }
    for (const text of ['make it winter', 'start over', 'make it night slowly', 'make it snow more']) {
      expect(finalizeDelay('typing', text)).toBe(2500);
      expect(finalizeDelay('burst', text)).toBe(600);
    }
  });
  it('waits for four fast keys, finalizes whole inserts, and resets after a final', () => {
    const input = makeInputClassifier();
    expect(input.classify('insertText', 'a', 100)).toBe('typing');
    expect(input.classify('insertText', 'b', 120)).toBe('typing');
    expect(input.classify('insertText', 'c', 140)).toBe('typing');
    expect(input.classify('insertText', 'd', 160)).toBe('burst');
    input.reset();
    expect(input.classify('insertText', 'e', 180)).toBe('typing');
    for (const type of ['insertFromPaste', 'insertFromDrop', 'insertReplacementText', 'insertFromYank']) {
      expect(input.classify(type, null, 200)).toBe('whole');
    }
    expect(input.classify('insertText', 'make it winter', 300)).toBe('whole');
  });
  it('keeps push-to-talk and reset independent of Wispr Ctrl and IME', () => {
    const key = { code: 'Backquote', ctrlKey: false, altKey: false, metaKey: false, shiftKey: false, isComposing: false };
    expect(isPushToTalk(key)).toBe(true);
    expect(isPushToTalk({ ...key, ctrlKey: true })).toBe(false);
    expect(isPushToTalk({ ...key, isComposing: true })).toBe(false);
    expect(isResetShortcut({ ...key, code: 'KeyR', altKey: true })).toBe(true);
    expect(isResetShortcut({ ...key, code: 'KeyR', altKey: true, ctrlKey: true })).toBe(false);
  });
});

describe('web engine parity', () => {
  it('retains server transcript and timing, and defaults optional strength', () => {
    expect(parseEngineState({ changes: [{ id: 'r1', label: 'river' }], heard: 'Add a RIVER', keyword: 'river',
      kind: 'final', fake: true, stamps: { decided: 2, shown: 210, invalid: 'x' }, t_client: 100, applies_from_chunk: 2 }))
      .toMatchObject({ modifiers: [{ id: 'r1', label: 'river', intensity: 1 }], heard: 'Add a RIVER', keyword: 'river',
        kind: 'final', fake: true, stamps: { decided: 2, shown: 210 }, tClient: 100, appliesFromChunk: 2 });
    expect(parseEngineState({ changes: [], t_client: Infinity, applies_from_chunk: -1 })).toMatchObject({ tClient: null, appliesFromChunk: null });
    expect(parseEngineState({ heard: 'unmatched', unmatched: true })?.heard).toBe('unmatched');
    expect(parseEngineState({ changes: [{ id: 'r', label: 2 }] })).toBeNull();
  });
  it('keeps a draft unconfirmed until submitted or cleared', () => {
    const heard = makeHeardTracker();
    heard.sentPartial('make it winter', 101, 99);
    expect(heard.classify('make it winter', 99)).toBe('pending');
    heard.sentFinal('make it winter', 300);
    expect(heard.classify('make it winter', 300)).toBe('confirmed');
    expect(heard.classify('add a lion', 301)).toBe('other');
    heard.sentPartial('add a lion', 501, 500);
    expect(heard.classify('add a lion', 501)).toBe('pending');
    expect(heard.classify('add a lion', 500)).toBe('pending');
    expect(heard.classify('add a lion', 777)).toBe('other');
    heard.sentPartial('make it night', 6001, 6000);
    heard.clearPartials();
    expect(heard.classify('make it night', 99)).toBe('other');
  });
  it('accepts web server aliases and uses secure sockets on HTTPS pages', () => {
    const loc = { protocol: 'https:', host: 'studio.test' };
    expect(engineBaseUrl('localhost:8100', loc)).toBe('wss://localhost:8100');
    expect(engineBaseUrl('https://engine.test/', loc)).toBe('wss://engine.test');
    expect(engineBaseUrl('wss://engine.test/', loc)).toBe('wss://engine.test');
    expect(engineBaseUrl(null, loc)).toBe('wss://studio.test');
    expect(() => engineBaseUrl('file:///tmp/private', loc)).toThrow();
    expect(() => engineBaseUrl('ws://engine.test', loc)).toThrow();
  });
  it('keeps video notices until both sockets and fresh frames are available', () => {
    const link = { controlOpen: false, framesOpen: false, lastFrameAt: null as number | null, everConnected: false };
    expect(videoNotice(link, 100)).toContain('Connecting');
    expect(videoNotice({ ...link, controlOpen: true, framesOpen: true }, 100)).toContain('first frame');
    expect(videoNotice({ ...link, controlOpen: true, framesOpen: true, lastFrameAt: 100 }, 2101)).toContain('paused');
    expect(videoNotice({ ...link, everConnected: true }, 5000)).toContain('Connection lost');
    expect(videoNotice({ ...link, controlOpen: true, framesOpen: true, lastFrameAt: 100 }, 200)).toBeNull();
  });
  it('correlates a frame already drawn before its control message, once', () => {
    const timing = makeLatencyTracker();
    timing.sent(100);
    expect(timing.onFrameDrawn(4, 120)).toBeNull();
    expect(timing.onFrameDrawn(5, 160)).toBeNull();
    expect(timing.onServer({ tClient: 100, appliesFromChunk: 5, unmatched: false, stamps: { shown: 50 } }))
      .toEqual({ ms: 60, stamps: { shown: 50 } });
    expect(timing.onServer({ tClient: 100, appliesFromChunk: 5, unmatched: false })).toBeNull();
  });
  it('measures each local final once and ignores unmatched or foreign command echoes', () => {
    const tracker = makeLatencyTracker();
    tracker.sent(100);
    tracker.onServer({ tClient: 100, appliesFromChunk: 5, unmatched: false, stamps: { decided: 2 } });
    expect(tracker.onFrameDrawn(4, 200)).toBeNull();
    expect(tracker.onFrameDrawn(5, 240)).toEqual({ ms: 140, stamps: { decided: 2 } });
    tracker.onServer({ tClient: 100, appliesFromChunk: 5, unmatched: false });
    expect(tracker.onFrameDrawn(6, 500)).toBeNull();
    tracker.sent(600);
    tracker.onServer({ tClient: 600, appliesFromChunk: 7, unmatched: true });
    expect(tracker.onFrameDrawn(7, 700)).toBeNull();
    tracker.onServer({ tClient: 999, appliesFromChunk: 7, unmatched: false });
    expect(tracker.onFrameDrawn(7, 1000)).toBeNull();
    tracker.reset();
  });
});
