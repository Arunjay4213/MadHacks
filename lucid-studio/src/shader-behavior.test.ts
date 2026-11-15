import { describe, expect, it } from 'vitest';
import { modifierTargets } from './components/WorldCanvas';
import { pulseAt } from './components/AcousticSingularity';

describe('bounded GPU inputs', () => {
  it('keeps modifiers independent and clamps bad intensities at the shader boundary', () => {
    const result = modifierTargets([
      { id: 'a', label: 'winter', intensity: 1 },
      { id: 'b', label: 'river', intensity: 5, coords: { x: 0.2, y: 0.6 } },
      { id: 'c', label: 'night', intensity: NaN },
      { id: 'd', label: 'lighthouse', intensity: 1 },
    ]);
    expect(result.t).toEqual({ winter: 1, river: 1.3, night: 0, sunset: 0, rain: 0, fog: 0 });
    expect(result.riverX).toBeCloseTo(-0.3);
  });
  it('implodes for 65ms then bursts and returns to rest', () => {
    expect(pulseAt(0)).toEqual({ scale: 1, burst: 0 });
    expect(pulseAt(64).scale).toBeLessThan(0.61);
    expect(pulseAt(65)).toEqual({ scale: 0.6, burst: 1 });
    expect(pulseAt(300).burst).toBeLessThan(1);
    expect(pulseAt(665)).toEqual({ scale: 1, burst: 0 });
    expect(pulseAt(NaN)).toEqual({ scale: 1, burst: 0 });
  });
});
