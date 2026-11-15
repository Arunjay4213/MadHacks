// Full-bleed world view. Default picture is a procedural alpine SIMULATION (not GPU model output);
// an external MediaStream or canvas, when given, replaces it under the same optical post pass.
import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { ActiveModifier, ModifierKind, MutationImpulse, SpatialCoords, WorldCanvasProps } from '../types';
import { fullscreenVert, worldFrag } from '../shaders/world';
import { postFrag } from '../shaders/post';

const WORLD_SCALE = 0.75; // the painted world is heavy; render it below screen resolution
const WORLD_MAX_PIXELS = 700_000; // hard cap on the procedural target, whatever the screen size
const EASE_RATE = 2.2; // per second, modifier cross-fades
const MAX_DPR = 1.5;

type ModKey = ModifierKind;
const KEYS: ModKey[] = ['winter', 'river', 'night', 'sunset', 'rain', 'fog'];
const UNIFORM = { winter: 'uWinter', river: 'uRiver', night: 'uNight', sunset: 'uSunset', rain: 'uRain', fog: 'uFog' } as const;

/** Strongest intensity per label (0..1.3), plus the river's horizontal position if given. */
export function modifierTargets(mods: ActiveModifier[]) {
  const t: Record<ModKey, number> = { winter: 0, river: 0, night: 0, sunset: 0, rain: 0, fog: 0 };
  let riverX = 0;
  for (const m of mods) {
    if (!(KEYS as string[]).includes(m.label)) continue; // labels the painted world cannot show
    const label = m.label as ModKey;
    const v = Number.isFinite(m.intensity) ? Math.min(1.3, Math.max(0, m.intensity)) : 0;
    if (v > t[label]) {
      t[label] = v;
      if (label === 'river' && m.coords) riverX = Math.min(0.5, Math.max(-0.5, m.coords.x - 0.5));
    }
  }
  return { t, riverX };
}

const quad = () => new THREE.PlaneGeometry(2, 2);

function Scene({ modifiers, source, impulse, pointer, reducedMotion, active, onReady }: WorldCanvasProps) {
  const { gl, size, viewport, invalidate } = useThree();
  const camera = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), []);
  const placeholder = useMemo(() => {
    const t = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    t.needsUpdate = true;
    return t;
  }, []);

  const world = useMemo(() => {
    const material = new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: worldFrag,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uRes: { value: new THREE.Vector2(1, 1) },
        uTilt: { value: new THREE.Vector2() },
        uWinter: { value: 0 },
        uRiver: { value: 0 },
        uNight: { value: 0 },
        uSunset: { value: 0 },
        uRain: { value: 0 },
        uFog: { value: 0 },
        uRiverX: { value: 0 },
      },
    });
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(quad(), material));
    return { scene, material };
  }, []);

  const target = useMemo(() => new THREE.WebGLRenderTarget(1, 1, { depthBuffer: false }), []);

  const post = useMemo(() => {
    const material = new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: postFrag,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uScene: { value: target.texture },
        uSource: { value: placeholder },
        uSourceMix: { value: 0 },
        uSourceAspect: { value: 16 / 9 },
        uRes: { value: new THREE.Vector2(1, 1) },
        uTime: { value: 0 },
        uImpulseAge: { value: 99 },
        uImpulseCenter: { value: new THREE.Vector2(0.5, 0.05) },
        uImpulseLocal: { value: 0 },
        uReduced: { value: 0 },
      },
    });
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(quad(), material));
    return { scene, material };
  }, [target, placeholder]);

  // Size the world target to the canvas (below native resolution).
  useEffect(() => {
    let fw = size.width * viewport.dpr * WORLD_SCALE;
    let fh = size.height * viewport.dpr * WORLD_SCALE;
    const cap = Math.min(1, Math.sqrt(WORLD_MAX_PIXELS / Math.max(1, fw * fh)));
    fw *= cap;
    fh *= cap;
    const w = Math.max(1, Math.round(fw));
    const h = Math.max(1, Math.round(fh));
    target.setSize(w, h);
    world.material.uniforms.uRes.value.set(w, h);
    post.material.uniforms.uRes.value.set(size.width * viewport.dpr, size.height * viewport.dpr);
    invalidate();
  }, [size, viewport.dpr, target, world, post, invalidate]);

  // External source: a MediaStream through a muted inline <video>, or a canvas. The parent owns the stream.
  const sourceTex = useRef<THREE.Texture | null>(null);
  const sourceVideo = useRef<HTMLVideoElement | null>(null);
  const sourceCanvas = useRef<HTMLCanvasElement | null>(null);
  const uploadedVersion = useRef<string | undefined>(undefined);
  useEffect(() => {
    sourceCanvas.current = null;
    sourceVideo.current = null;
    uploadedVersion.current = undefined;
    if (!source) return;
    let tex: THREE.Texture;
    let video: HTMLVideoElement | null = null;
    const setAspect = (w: number, h: number) => {
      if (w > 0 && h > 0) post.material.uniforms.uSourceAspect.value = w / h;
    };
    if (source instanceof HTMLCanvasElement) {
      tex = new THREE.CanvasTexture(source);
      sourceCanvas.current = source;
      setAspect(source.width, source.height);
    } else {
      video = document.createElement('video');
      sourceVideo.current = video;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = source;
      const v = video;
      v.onloadedmetadata = () => { setAspect(v.videoWidth, v.videoHeight); invalidate(); };
      v.onloadeddata = () => invalidate();
      v.play().catch(() => {}); // autoplay may be blocked; muted inline video normally plays
      tex = new THREE.VideoTexture(v);
    }
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    sourceTex.current = tex;
    post.material.uniforms.uSource.value = tex;
    post.material.uniforms.uSourceMix.value = 1; // An actual source never starts as a procedural frame.
    invalidate();
    return () => {
      post.material.uniforms.uSource.value = placeholder;
      post.material.uniforms.uSourceMix.value = 0;
      sourceVideo.current = null;
      sourceTex.current = null;
      sourceCanvas.current = null;
      tex.dispose();
      if (video) {
        video.pause();
        video.srcObject = null; // detach only; never stop tracks the parent owns
      }
    };
  }, [source, post, placeholder, invalidate]);

  // Dispose GPU resources on unmount.
  useEffect(
    () => () => {
      world.material.dispose();
      post.material.dispose();
      world.scene.traverse((o) => o instanceof THREE.Mesh && o.geometry.dispose());
      post.scene.traverse((o) => o instanceof THREE.Mesh && o.geometry.dispose());
      target.dispose();
      placeholder.dispose();
    },
    [world, post, target, placeholder],
  );

  const goal = useMemo(() => modifierTargets(modifiers), [modifiers]);
  const demand = !!reducedMotion || active === false; // matches the Canvas frameloop below
  const latest = useRef({ goal, pointer, impulse, reducedMotion: !!reducedMotion, demand, onReady });
  latest.current = { goal, pointer, impulse, reducedMotion: !!reducedMotion, demand, onReady };
  useEffect(() => invalidate(), [goal, pointer, impulse, reducedMotion, active, invalidate]);

  const time = useRef(0);
  const tilt = useRef(new THREE.Vector2());
  const tiltGoal = useRef(new THREE.Vector2());
  const ready = useRef(false);

  useFrame((_, rawDt) => {
    const { goal: g, pointer: p, impulse: imp, reducedMotion: still, demand: onDemand, onReady: ready_ } = latest.current;
    if (active === false && ready.current) return;
    if (source && !sourceTex.current) { invalidate(); return; }
    const dt = Math.min(rawDt, 0.1);
    const u = world.material.uniforms;
    const k = still ? 1 : 1 - Math.exp(-dt * EASE_RATE);
    let moving = false;
    for (const key of KEYS) {
      const uni = u[UNIFORM[key]];
      const next = uni.value + (g.t[key] - uni.value) * k;
      if (Math.abs(next - g.t[key]) > 0.002) moving = true;
      uni.value = Math.abs(next - g.t[key]) < 0.0005 ? g.t[key] : next;
    }
    u.uRiverX.value += (g.riverX - u.uRiverX.value) * k;

    if (!still) time.current += dt;
    u.uTime.value = time.current;
    const tx = still ? 0 : (clamp01(p.x) - 0.5) * 2;
    const ty = still ? 0 : (0.5 - clamp01(p.y)) * 2;
    tiltGoal.current.set(tx, ty);
    tilt.current.lerp(tiltGoal.current, still ? 1 : 1 - Math.exp(-dt * 3));
    u.uTilt.value.copy(tilt.current);

    const pu = post.material.uniforms;
    pu.uTime.value = time.current;
    pu.uReduced.value = still ? 1 : 0;
    const mixGoal = sourceTex.current ? 1 : 0;
    const mixNext = pu.uSourceMix.value + (mixGoal - pu.uSourceMix.value) * (still ? 1 : 1 - Math.exp(-dt * 4));
    pu.uSourceMix.value = Math.abs(mixGoal - mixNext) < 0.0005 ? mixGoal : mixNext;
    const cnv = sourceCanvas.current;
    if (cnv && sourceTex.current) {
      // Engine canvases mark each decoded frame in dataset.frameVersion: upload only when it changes.
      // A generic canvas has no marker and is uploaded every frame.
      const version = cnv.dataset.frameVersion;
      if (version === undefined || version !== uploadedVersion.current) {
        uploadedVersion.current = version;
        sourceTex.current.needsUpdate = true;
        if (cnv.width > 0 && cnv.height > 0) pu.uSourceAspect.value = cnv.width / cnv.height;
      }
    }
    const age = imp ? (performance.now() - imp.startedAt) / 1000 : 99;
    pu.uImpulseAge.value = age;
    setCenter(pu.uImpulseCenter.value, imp);
    pu.uImpulseLocal.value = imp?.localized && imp.center ? 1 : 0;

    // A fully faded-in external source hides the world completely: skip its (expensive) pass.
    if (pu.uSourceMix.value < 0.999) {
      gl.setRenderTarget(target);
      gl.render(world.scene, camera);
      gl.setRenderTarget(null);
    }
    gl.render(post.scene, camera);

    if (!ready.current && (!source || sourceTex.current) && (!sourceVideo.current || sourceVideo.current.readyState >= 2)) {
      // One-time poster of the first real frame, read right after drawing (no preserveDrawingBuffer needed).
      ready.current = true;
      let preview: string | undefined;
      try {
        preview = gl.domElement.toDataURL('image/jpeg', 0.85);
      } catch {
        preview = undefined;
      }
      ready_?.(preview);
    }
    // In demand mode keep rendering while something is still settling or a video is playing.
    if (onDemand && active !== false && (moving || sourceTex.current)) invalidate();
  }, 1);

  return null;
}

function clamp01(v: number) {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5;
}

function setCenter(out: THREE.Vector2, imp: MutationImpulse | null) {
  const c: SpatialCoords | undefined = imp?.localized ? imp.center : undefined;
  // Coords are top-left origin; GL is bottom-left. Unlocalized commits ripple up from the bottom centre (the voice ring).
  out.set(c ? clamp01(c.x) : 0.5, c ? 1 - clamp01(c.y) : 0.05);
}

export function WorldCanvas(props: WorldCanvasProps) {
  const still = !!props.reducedMotion;
  const demand = still || props.active === false; // inactive or reduced motion: render only on change
  return (
    <Canvas
      linear
      flat
      dpr={[1, still ? 1 : MAX_DPR]}
      frameloop={demand ? 'demand' : 'always'}
      gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block', background: '#05070a' }}
      aria-hidden
    >
      <Scene {...props} />
    </Canvas>
  );
}

export default WorldCanvas;
