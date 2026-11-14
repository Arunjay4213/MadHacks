import type { ActiveModifier, ModifierKind, SpatialCoords } from './types';

export type WorldIntent =
  | { kind: 'add'; label: ModifierKind; intensity: number; coords?: SpatialCoords }
  | { kind: 'remove'; label: string }
  | { kind: 'intensity'; id: string; intensity: number }
  | { kind: 'clear' | 'undo' };
export interface WorldState {
  modifiers: ActiveModifier[];
  history: ActiveModifier[][];
  nextId: number;
}
export const emptyWorld = (): WorldState => ({ modifiers: [], history: [], nextId: 1 });
export const clampIntensity = (n: number) => Number.isFinite(n) ? Math.min(1.3, Math.max(0, n)) : 1;
export function normalizeCoords(coords: SpatialCoords): SpatialCoords | null {
  if (!Number.isFinite(coords.x) || !Number.isFinite(coords.y)) return null;
  return { x: Math.max(0, Math.min(1, coords.x)), y: Math.max(0, Math.min(1, coords.y)) };
}

export function parseIntent(prompt: string, current: ActiveModifier[]): WorldIntent | null {
  const text = prompt.toLowerCase().replace(/[‘’]/g, "'");
  if (/\b(?:undo|go back|never mind)\b/.test(text)) return { kind: 'undo' };
  if (/\b(?:don't|do not|not|into|instead of)\b/.test(text)) return null;
  if (/\b(?:clear everything|clear all|reset|start over)\b/.test(text)) return { kind: 'clear' };
  if (/\b(?:summer|daytime|daylight)\b/.test(text)) {
    return { kind: 'remove', label: /summer/.test(text) ? 'winter' : 'night' };
  }
  const match = text.match(/\b(winter|snow|snowy|river|stream|night|nighttime|sunset|rain|raining|rainy|fog|foggy|mist)\b/);
  if (!match) return null;
  const aliases: Record<string, ModifierKind> = {
    snow: 'winter', snowy: 'winter', stream: 'river', nighttime: 'night', raining: 'rain', rainy: 'rain', foggy: 'fog', mist: 'fog',
  };
  const label = aliases[match[1]] ?? match[1] as ModifierKind;
  if (/\b(?:remove|without|no more|clear|delete|get rid of)\b/.test(text)) return { kind: 'remove', label };
  const existing = current.find(m => m.label === label);
  const intensity = /\b(?:a little|slightly|a bit|subtle)\b/.test(text) ? 0.3
    : /\b(?:a lot|heavy|much more)\b/.test(text) ? 1.3
    : /\bmore\b/.test(text) ? clampIntensity((existing?.intensity ?? 0.8) + 0.2)
    : /\bless\b/.test(text) ? clampIntensity((existing?.intensity ?? 1) - 0.2) : 1;
  return { kind: 'add', label, intensity };
}

export function evolve(state: WorldState, intent: WorldIntent): WorldState {
  if (intent.kind === 'undo') {
    if (!state.history.length) return state;
    return { ...state, modifiers: state.history.at(-1)!, history: state.history.slice(0, -1) };
  }
  let modifiers = state.modifiers;
  let nextId = state.nextId;
  if (intent.kind === 'clear') modifiers = [];
  if (intent.kind === 'remove') modifiers = modifiers.filter(m => m.label !== intent.label);
  if (intent.kind === 'intensity') modifiers = modifiers.map(m => m.id === intent.id
    ? { ...m, intensity: clampIntensity(intent.intensity) } : m);
  if (intent.kind === 'add') {
    const previous = modifiers.find(m => m.label === intent.label);
    const conflicts = intent.label === 'night' ? 'sunset' : intent.label === 'sunset' ? 'night' : null;
    modifiers = modifiers.filter(m => m.label !== intent.label && m.label !== conflicts);
    const modifier: ActiveModifier = {
      id: previous?.id ?? `m${nextId++}`, label: intent.label, intensity: clampIntensity(intent.intensity),
      ...(previous?.coords ? { coords: previous.coords } : {}),
      ...(intent.coords ? { coords: normalizeCoords(intent.coords) ?? undefined } : {}),
    };
    const at = previous ? state.modifiers.indexOf(previous) : modifiers.length;
    modifiers = [...modifiers.slice(0, at), modifier, ...modifiers.slice(at)];
  }
  if (JSON.stringify(modifiers) === JSON.stringify(state.modifiers)) return state;
  return { modifiers, nextId, history: [...state.history, state.modifiers].slice(-100) };
}
