import { describe, expect, it } from 'vitest';
import { TranscriptGate, splitBands } from './input';
import { parseFrame, parseEngineState } from './engine';

describe('input boundaries', () => {
  it('deduplicates interim revisions and committed speech per recognition session', () => {
    const gate = new TranscriptGate();
    expect(gate.accept(0, 'make it win', false)).toBe('partial');
    expect(gate.accept(0, 'make it win', false)).toBeNull();
    expect(gate.accept(0, 'make it winter', false)).toBe('partial');
    expect(gate.accept(0, 'make it winter', true)).toBe('final');
    expect(gate.accept(0, 'make it winter', true)).toBeNull();
    gate.reset();
    expect(gate.accept(0, 'make it winter', true)).toBe('final');
  });
  it('splits the actual sample-rate bins into vocal, low and sibilance bands', () => {
    const data = new Uint8Array(256);
    data.fill(255, 4, 32);
    const bands = splitBands(data, 48000, 512);
    expect(bands.low).toBe(0);
    expect(bands.vocal).toBeGreaterThan(0.9);
    expect(bands.high).toBe(0);
  });
  it('reads the agreed six-byte chunk/frame header and rejects bad frames', () => {
    const data = new Uint8Array([0, 0, 1, 4, 0, 3, 255, 216, 255, 217]).buffer;
    expect(parseFrame(data)).toMatchObject({ chunk: 260, index: 3 });
    expect(parseFrame(new ArrayBuffer(5))).toBeNull();
    expect(parseFrame(new ArrayBuffer(20))).toBeNull();
  });
  it('keeps arbitrary server modifier labels and rejects malformed states', () => {
    const state = parseEngineState({ changes: [{ id: 'c1', label: 'lighthouse', strength: 0.4 }], unmatched: false });
    expect(state?.modifiers).toEqual([{ id: 'c1', label: 'lighthouse', intensity: 0.4 }]);
    expect(parseEngineState({ changes: [{ id: 1 }] })).toBeNull();
  });
});
