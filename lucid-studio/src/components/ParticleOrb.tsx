import { useEffect, useRef } from 'react';

interface Props {
  level: number;        // 0..1, how loud the voice is right now
  listening: boolean;   // the microphone is on
  reducedMotion: boolean;
  size?: number;
}

const COUNT = 520;

// A sphere of particles that breathes when idle and swells with the voice while listening.
export default function ParticleOrb({ level, listening, reducedMotion, size = 116 }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const state = useRef({ level, listening });
  state.current = { level, listening };

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = size * dpr; el.height = size * dpr;
    ctx.scale(dpr, dpr);
    // Points spread evenly on a sphere (Fibonacci lattice), each with its own phase.
    const golden = Math.PI * (3 - Math.sqrt(5));
    const points = Array.from({ length: COUNT }, (_, i) => {
      const y = 1 - (i / (COUNT - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const theta = golden * i;
      return { x: Math.cos(theta) * r, y, z: Math.sin(theta) * r, phase: (i * 0.618) % (Math.PI * 2) };
    });
    let smooth = 0;
    let frame = 0;
    const draw = (time: number) => {
      const t = time / 1000;
      const { level: lv, listening: on } = state.current;
      smooth += ((on ? lv : 0) - smooth) * 0.18;
      const c = size / 2;
      const base = size * (on ? 0.33 : 0.3) * (1 + (reducedMotion ? 0 : 0.03 * Math.sin(t * 1.6)));
      const spin = reducedMotion ? 0.6 : t * (on ? 0.9 : 0.35);
      const cos = Math.cos(spin), sin = Math.sin(spin);
      ctx.clearRect(0, 0, size, size);
      if (on) {
        const glow = ctx.createRadialGradient(c, c, base * 0.2, c, c, base * 1.6);
        glow.addColorStop(0, `rgba(240, 150, 90, ${0.18 + smooth * 0.35})`);
        glow.addColorStop(1, 'rgba(240, 150, 90, 0)');
        ctx.fillStyle = glow; ctx.fillRect(0, 0, size, size);
      }
      for (const p of points) {
        const x = p.x * cos - p.z * sin;
        const z = p.x * sin + p.z * cos;
        const wobble = reducedMotion ? 0 : Math.sin(t * 6 + p.phase * 3) * smooth * 0.35 + Math.sin(t * 2 + p.phase) * 0.02;
        const r = base * (1 + wobble + smooth * 0.25);
        const depth = (z + 1) / 2;
        const px = c + x * r;
        const py = c + p.y * r;
        const dot = 0.6 + depth * 1.3;
        ctx.globalAlpha = 0.25 + depth * 0.75;
        ctx.fillStyle = on ? `rgb(${245}, ${150 + Math.round(depth * 70)}, ${90 + Math.round(depth * 60)})` : `rgb(${170 + Math.round(depth * 60)}, ${190 + Math.round(depth * 45)}, ${178 + Math.round(depth * 40)})`;
        ctx.beginPath(); ctx.arc(px, py, dot, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [size, reducedMotion]);

  return <canvas ref={canvas} style={{ width: size, height: size, display: 'block' }} aria-hidden="true" />;
}
