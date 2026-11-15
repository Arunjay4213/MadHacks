import { describe, expect, it } from 'vitest';
import { emptyWorld, evolve, parseIntent, normalizeCoords } from './world';

describe('compositional world', () => {
  it('stacks snow, river and night and removes only the river', () => {
    let state = emptyWorld();
    for (const text of ['make it snow', 'also add a river', 'turn to night']) {
      state = evolve(state, parseIntent(text, state.modifiers)!);
    }
    expect(state.modifiers.map(m => m.label)).toEqual(['winter', 'river', 'night']);
    state = evolve(state, parseIntent('remove the river', state.modifiers)!);
    expect(state.modifiers.map(m => m.label)).toEqual(['winter', 'night']);
    state = evolve(state, { kind: 'undo' });
    expect(state.modifiers.map(m => m.label)).toEqual(['winter', 'river', 'night']);
  });
  it('replaces lighting without destroying weather and retains stable ids', () => {
    let state = evolve(emptyWorld(), { kind: 'add', label: 'winter', intensity: 1 });
    const winter = state.modifiers[0].id;
    state = evolve(state, { kind: 'add', label: 'night', intensity: 1 });
    state = evolve(state, { kind: 'add', label: 'sunset', intensity: 1 });
    expect(state.modifiers.map(m => m.label)).toEqual(['winter', 'sunset']);
    expect(state.modifiers[0].id).toBe(winter);
  });
  it('clear is undoable, and no-op removals do not consume history', () => {
    let state = evolve(emptyWorld(), { kind: 'add', label: 'river', intensity: 1 });
    state = evolve(state, { kind: 'clear' });
    const unchanged = evolve(state, { kind: 'remove', label: 'winter' });
    expect(unchanged).toBe(state);
    expect(evolve(unchanged, { kind: 'undo' }).modifiers[0].label).toBe('river');
  });
  it('ignores negation and unknown commands instead of guessing', () => {
    for (const text of ['don’t add a river', 'do not make it winter', 'make the tree into a dragon', 'snowboard']) {
      expect(parseIntent(text, [])).toBeNull();
    }
    expect(parseIntent('remove snow', [])).toEqual({ kind: 'remove', label: 'winter' });
  });
  it('uses bounded intensity and stores normalized spatial coordinates', () => {
    const coords = normalizeCoords({ x: -1, y: 2 });
    expect(coords).toEqual({ x: 0, y: 1 });
    expect(normalizeCoords({ x: NaN, y: 0 })).toBeNull();
    let state = evolve(emptyWorld(), { kind: 'add', label: 'river', intensity: 1, coords: coords! });
    state = evolve(state, { kind: 'intensity', id: state.modifiers[0].id, intensity: 9 });
    expect(state.modifiers[0].intensity).toBe(1.3);
    expect(state.modifiers[0].coords).toEqual(coords);
    expect(parseIntent('a little snow', [])).toMatchObject({ intensity: 0.3 });
  });
});
