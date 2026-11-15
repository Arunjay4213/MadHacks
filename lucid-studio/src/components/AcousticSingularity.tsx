// Voice-reactive filament ring: 6000 additive points on its own transparent canvas.
// The parent positions it, animates its visibility and supplies bands (real or oscillator fallback).
import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { AcousticSingularityProps, AudioBands, MutationImpulse } from '../types';
import { singularityFrag, singularityVert } from '../shaders/singularity';

const COUNT = 6000;
const IMPLODE_MS = 65;
const BURST_MS = 600;
const MAX_DPR = 1.5;

/** Ring scale and burst strength at `ageMs` after a commit impulse. */
export function pulseAt(ageMs: number): { scale: number; burst: number } {
  if (!(ageMs >= 0) || ageMs >= IMPLODE_MS + BURST_MS) return { scale: 1, burst: 0 };
  if (ageMs < IMPLODE_MS) return { scale: 1 - 0.4 * (ageMs / IMPLODE_MS), burst: 0 };
  const t = (ageMs - IMPLODE_MS) / BURST_MS;
  const out = 1 - Math.pow(1 - t, 3); // ease-out cubic
  return { scale: 0.6 + 0.4 * out, burst: 1 - t };
}

const clampBand = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

function Ring({ bands, impulse, reducedMotion }: AcousticSingularityProps) {
  const { viewport } = useThree();
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(COUNT * 3); // real positions come from the shader
    const angle = new Float32Array(COUNT);
    const seed = new Float32Array(COUNT);
    const lane = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      angle[i] = (i / COUNT) * Math.PI * 2 + Math.random() * 0.002;
      seed[i] = Math.random();
      lane[i] = Math.random();
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aAngle', new THREE.BufferAttribute(angle, 1));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    g.setAttribute('aLane', new THREE.BufferAttribute(lane, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4); // shader-displaced: skip culling surprises
    return g;
  }, []);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: singularityVert,
        fragmentShader: singularityFrag,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uLow: { value: 0 },
          uVocal: { value: 0 },
          uHigh: { value: 0 },
          uScale: { value: 1 },
          uBurst: { value: 0 },
          uPixelRatio: { value: 1 },
        },
      }),
    [],
  );

  useEffect(() => () => {
    geometry.dispose();
    material.dispose();
  }, [geometry, material]);

  const latest = useRef<{ bands: AudioBands; impulse: MutationImpulse | null; still: boolean }>({
    bands,
    impulse,
    still: !!reducedMotion,
  });
  latest.current = { bands, impulse, still: !!reducedMotion };
  const smooth = useRef({ low: 0, vocal: 0, high: 0 });
  const time = useRef(0);

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.1);
    const { bands: b, impulse: imp, still } = latest.current;
    const u = material.uniforms;
    // Fast attack, slower release, so the ring breathes instead of flickering.
    for (const key of ['low', 'vocal', 'high'] as const) {
      const target = clampBand(b[key]);
      const cur = smooth.current[key];
      const rate = target > cur ? 18 : 4;
      smooth.current[key] = cur + (target - cur) * (1 - Math.exp(-dt * rate));
    }
    // Reduced motion: no flow or geometry pulse; the ring only brightens with the voice.
    const motion = still ? 0.25 : 1;
    if (!still) time.current += dt;
    u.uTime.value = time.current;
    u.uLow.value = smooth.current.low * motion;
    u.uVocal.value = smooth.current.vocal * motion;
    u.uHigh.value = smooth.current.high * motion;
    const { scale, burst } = imp ? pulseAt(performance.now() - imp.startedAt) : { scale: 1, burst: 0 };
    u.uScale.value = still ? 1 : scale;
    u.uBurst.value = still ? 0 : burst;
    u.uPixelRatio.value = viewport.dpr;
  });

  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

export function AcousticSingularity(props: AcousticSingularityProps) {
  return (
    <Canvas
      linear
      flat
      dpr={[1, MAX_DPR]}
      camera={{ position: [0, 0, 2.7], fov: 42 }}
      gl={{ antialias: false, alpha: true, premultipliedAlpha: true, powerPreference: 'high-performance' }}
      style={{ width: '100%', height: '100%', display: 'block', background: 'transparent', pointerEvents: 'none' }}
      aria-hidden
    >
      <Ring {...props} />
    </Canvas>
  );
}

export default AcousticSingularity;
