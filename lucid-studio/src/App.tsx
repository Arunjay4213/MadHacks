import { Component, lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import ThresholdShell from './components/ThresholdShell';
import WhisperCaption from './components/WhisperCaption';
import LucidOverlay from './components/LucidOverlay';
import VoiceAgent, { type AudioControls } from './components/VoiceAgent';
import CommandInput from './components/CommandInput';
import { useGenerativeWorld } from './hooks/useGenerativeWorld';
import { useEngineStream } from './engine';
import type { ActiveModifier, WorldStreamAdapter } from './types';

const WorldCanvas = lazy(() => import('./components/WorldCanvas'));

class OpticalBoundary extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch() { this.props.onError(); }
  render() {
    return this.state.failed ? <div className="optical-fallback" role="status">
      <p>The light needs a little more room.</p><span>WebGL is unavailable. Try a hardware-accelerated browser; keyboard commands still work.</span>
    </div> : this.props.children;
  }
}

interface StudioProps {
  adapter?: WorldStreamAdapter;
  source?: MediaStream | HTMLCanvasElement | null;
  remoteModifiers?: ActiveModifier[];
  initialPrompt?: string;
}

export default function App({ adapter, source, remoteModifiers, initialPrompt }: StudioProps) {
  const reducedMotion = Boolean(useReducedMotion());
  const [studio, setStudio] = useState(false);
  const [editorVisible, setEditorVisible] = useState(false);
  const [preview, setPreview] = useState<string>();
  const [ready, setReady] = useState(false);
  const [inputOpen, setInputOpen] = useState(false);
  const [heldMemory, setHeldMemory] = useState(false);
  const [hoverMemory, setHoverMemory] = useState(false);
  const [pinnedMemory, setPinnedMemory] = useState(false);
  const [help, setHelp] = useState(false);
  const [notice, setNotice] = useState('');
  const noticeTimer = useRef<ReturnType<typeof setTimeout>>();
  const audioApi = useRef<AudioControls>();
  const notify = useCallback((message: string) => {
    setNotice(message); clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(''), 4800);
  }, []);
  const endpoint = useRef(new URLSearchParams(window.location.search).get('engine')
    ?? (window.location.pathname.startsWith('/studio') ? window.location.origin : null)).current;
  const live = Boolean(endpoint || adapter || source);
  const engine = useEngineStream(endpoint, notify);
  const world = useGenerativeWorld({
    adapter: endpoint ? engine.adapter : adapter,
    remoteModifiers: endpoint ? engine.remoteState?.modifiers ?? [] : remoteModifiers,
    partial: endpoint ? text => engine.sendText('partial', text) : undefined,
    tap: endpoint ? engine.tap : undefined,
    notify, requestAudio: async () => audioApi.current ? audioApi.current.request() : false,
    stopAudio: () => audioApi.current?.stop(),
  });
  const pointer = useRef({ x: 0.5, y: 0.5 });
  const registerAudio = useCallback((api: AudioControls) => { audioApi.current = api; }, []);
  const originalPrompt = initialPrompt ?? (live ? 'A wide shot of a meadow with a single oak tree.'
    : 'A quiet alpine valley. Mountain silhouettes, drifting mist, and the soft light of a new day.');
  const worldReady = useCallback((image?: string) => { setReady(true); if (image) setPreview(image); }, []);
  const opticalError = useCallback(() => setReady(true), []);
  const enter = useCallback(() => { setStudio(true); }, []);
  useEffect(() => () => clearTimeout(noticeTimer.current), []);
  useEffect(() => { if (!editorVisible) world.stopVoice(); }, [editorVisible, world.stopVoice]);

  useEffect(() => {
    if (!studio || !editorVisible) return;
    const down = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && (['INPUT', 'TEXTAREA'].includes(event.target.tagName) || event.target.isContentEditable);
      if (event.key === 'Escape') { setInputOpen(false); setHelp(false); setPinnedMemory(false); return; }
      if (typing) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); world.undo(); return; }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === 'Tab' && !pinnedMemory && !(event.target instanceof HTMLButtonElement)) {
        event.preventDefault(); setHeldMemory(true); return;
      }
      if (event.repeat) return;
      if (event.key === '/') { event.preventDefault(); setInputOpen(true); setHelp(false); }
      if (event.key === '?' || event.key === 'F1') { event.preventDefault(); setHelp(value => !value); setInputOpen(false); }
      if (event.key.toLowerCase() === 'm') setPinnedMemory(value => !value);
      if (event.key.toLowerCase() === 'v') void world.startVoice();
      if (event.key === 'Backspace') { event.preventDefault(); world.undo(); }
      if (event.key === '0') world.reset();
      const examples: Record<string, string> = { '1': 'make it winter', '2': 'also add a river', '3': 'turn to night', '4': 'make it sunset', '5': 'let it rain' };
      if (examples[event.key]) { setHelp(false); void world.commit(examples[event.key]); }
    };
    const up = (event: KeyboardEvent) => { if (event.key === 'Tab') setHeldMemory(false); };
    const blur = () => setHeldMemory(false);
    window.addEventListener('keydown', down); window.addEventListener('keyup', up); window.addEventListener('blur', blur);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); window.removeEventListener('blur', blur); };
  }, [studio, editorVisible, pinnedMemory, world.undo, world.reset, world.commit, world.startVoice]);

  const panel = <><div className="prompt-scroll" data-lenis-prevent>
    <p className="eyebrow">THE STARTING POINT</p><h3>Original prompt</h3>
    <p className="original-prompt">{originalPrompt}</p>
    <div className="prompt-provenance"><span />{live ? (engine.connected ? 'Live engine connected' : 'Waiting for the engine') : 'Procedural world simulation'}</div>
    <div className="scene-composition"><div className="composition-heading"><span>SCENE MEMORY</span><span>{String(world.activeModifiers.length).padStart(2, '0')}</span></div>
      {!world.activeModifiers.length && <p className="untouched-note">Nothing added. Everything is possible.</p>}
      {world.activeModifiers.map(modifier => <div className="editor-modifier" key={modifier.id}>
        <span>{modifier.label}</span>{live ? <div className="live-strength">
          <button aria-label={`Less ${modifier.label}`} onClick={() => world.scrub(modifier, -0.1)}>−</button>
          <span>{Math.round(modifier.intensity * 100)}%</span><button aria-label={`More ${modifier.label}`} onClick={() => world.scrub(modifier, 0.1)}>+</button>
        </div> : <input type="range" min="0" max="1.3" step="0.05" value={modifier.intensity} aria-label={`${modifier.label} intensity`}
          onChange={event => world.scrub(modifier, Number(event.target.value) - modifier.intensity)} />}
        <button aria-label={`Remove ${modifier.label}`} onClick={() => world.remove(modifier)}>×</button>
      </div>)}
      <div className="composition-actions"><button onClick={world.undo}>↶ Undo</button><button onClick={world.reset}>Reset scene</button></div>
    </div>
    <CommandInput compact commit={text => void world.commit(text)} />
    <p className="editor-hint">Try <button onClick={() => void world.commit('make it winter')}>winter</button>, then <button onClick={() => void world.commit('also add a river')}>a river</button>.<br/>Your next thought builds on the last.</p>
    <button className="gesture-link" onClick={() => setHelp(true)}>Keyboard & voice gestures ↗</button>
    {live && engine.latency !== null && <p className="editor-latency">{engine.latency} ms <span>TO FIRST CORRELATED FRAME</span></p>}
  </div><div className="voice-dock"><VoiceAgent visible={editorVisible} listening={world.listening}
    simulateUntil={world.simulateUntil} impulse={world.impulse} reducedMotion={reducedMotion}
    register={registerAudio} toggle={() => void world.startVoice()} /></div></>;

  return <><ThresholdShell reducedMotion={reducedMotion} onEnter={enter} onVisibility={setEditorVisible} live={live} preview={preview} panel={panel}>
    <div className="living-viewport" data-mode={live ? 'live' : 'simulation'}
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect();
        pointer.current.x = (event.clientX - rect.left) / rect.width; pointer.current.y = (event.clientY - rect.top) / rect.height;
      }}>
      <OpticalBoundary onError={opticalError}><Suspense fallback={null}><WorldCanvas modifiers={world.activeModifiers} source={source ?? engine.source}
        impulse={world.impulse} pointer={pointer.current} reducedMotion={reducedMotion} active={editorVisible} onReady={worldReady} /></Suspense></OpticalBoundary>
      {!ready && <div className="assembling-light" aria-live="polite">Gathering light…</div>}
      {endpoint && !engine.source && ready && <p className="engine-awaiting" role="status">Waiting for the live world…</p>}
      {studio && <>
        <div className="touch-surface" aria-label="Click the landscape to place your next change" onPointerDown={event => {
          if (event.button !== 0) return;
          const rect = event.currentTarget.getBoundingClientRect(); world.placePin({ x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height });
        }} />
        {!inputOpen && <WhisperCaption text={world.transcript} interim={world.interim} reducedMotion={reducedMotion} />}
        <LucidOverlay pin={world.pin} memory={heldMemory || hoverMemory || pinnedMemory} hoverMemory={setHoverMemory}
          modifiers={world.activeModifiers} remove={world.remove} scrub={world.scrub} inputOpen={inputOpen}
          closeInput={() => setInputOpen(false)} commit={text => void world.commit(text)} help={help} closeHelp={() => setHelp(false)}
          live={live} connected={endpoint ? engine.connected : Boolean(source)} listening={world.listening} latency={engine.latency} undo={world.undo} reset={world.reset} />
        <AnimatePresence>{notice && <motion.p className="optical-notice" role="status" key={notice}
          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>{notice}</motion.p>}</AnimatePresence>
        <div className="sr-only" aria-live="polite">{world.activeModifiers.length ? `Active changes: ${world.activeModifiers.map(m => m.label).join(', ')}` : 'An untouched landscape.'}</div>
      </>}
    </div>
  </ThresholdShell>
  </>;
}
