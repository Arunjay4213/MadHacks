import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import ThresholdShell from './components/ThresholdShell';
import WhisperCaption from './components/WhisperCaption';
import LucidOverlay from './components/LucidOverlay';
import VoiceAgent, { type AudioControls } from './components/VoiceAgent';
import CommandInput from './components/CommandInput';
import OpticalBoundary from './components/OpticalBoundary';
import { useGenerativeWorld } from './hooks/useGenerativeWorld';
import { useEngineStream } from './engine';
import { isPushToTalk, isResetShortcut } from './input';
import type { ActiveModifier, WorldStreamAdapter } from './types';

const WorldCanvas = lazy(() => import('./components/WorldCanvas'));

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
  const params = new URLSearchParams(window.location.search);
  const endpoint = useRef(params.get('engine') || params.get('server')
    || (window.location.pathname.startsWith('/studio') ? window.location.origin : null)).current;
  const live = Boolean(endpoint || source);
  const engine = useEngineStream(endpoint, notify);
  const world = useGenerativeWorld({
    adapter: endpoint ? engine.adapter : adapter,
    remoteModifiers: endpoint ? engine.remoteState?.modifiers ?? [] : remoteModifiers,
    partial: endpoint ? text => engine.sendText('partial', text) : undefined,
    tap: endpoint ? engine.tap : undefined,
    notify, requestAudio: async () => audioApi.current ? audioApi.current.request() : false,
    stopAudio: () => audioApi.current?.stop(),
  });
  const disabled = Boolean(endpoint && !engine.connected);
  const effectiveSource = source ?? engine.source;
  const sourceLabel = endpoint ? (engine.remoteState?.fake ? 'CPU FAKE ENGINE · TEST VIDEO' : 'ENGINE STREAM')
    : live ? 'EXTERNAL VIDEO SOURCE' : 'PROCEDURAL STUDY';
  const [words, setWords] = useState({ text: '', keyword: '', unmatched: false, kind: 'local' });
  const localCommit = useCallback((text: string) => {
    setWords({ text, keyword: '', unmatched: false, kind: 'final' });
    return world.commit(text);
  }, [world.commit]);
  const localPartial = useCallback((text: string) => {
    setWords({ text, keyword: '', unmatched: false, kind: 'partial' });
    world.partial(text);
  }, [world.partial]);
  useEffect(() => {
    if (world.transcript) setWords({ text: world.transcript, keyword: '', unmatched: false, kind: world.interim ? 'partial' : 'final' });
  }, [world.transcript, world.interim]);
  const heard = engine.remoteState?.heard;
  const keyword = engine.remoteState?.keyword ?? '';
  const unmatched = engine.remoteState?.unmatched ?? false;
  const kind = engine.remoteState?.kind ?? 'server';
  const tClient = engine.remoteState?.tClient;
  useEffect(() => { if (heard !== null && heard !== undefined) setWords({ text: heard, keyword, unmatched, kind }); }, [heard, keyword, unmatched, kind, tClient]);
  const keywordAt = words.keyword ? words.text.toLowerCase().indexOf(words.keyword.toLowerCase()) : -1;
  const pointer = useRef({ x: 0.5, y: 0.5 });
  const registerAudio = useCallback((api: AudioControls) => { audioApi.current = api; }, []);
  const originalPrompt = initialPrompt ?? (live ? 'A wide shot of a meadow with a single oak tree.'
    : 'A quiet alpine valley. Mountain silhouettes, drifting mist, and the soft light of a new day.');
  const worldReady = useCallback((image?: string) => { setReady(true); if (image) setPreview(image); }, []);
  const opticalError = useCallback(() => setReady(true), []);
  const enter = useCallback(() => { setStudio(true); }, []);
  useEffect(() => () => clearTimeout(noticeTimer.current), []);
  useEffect(() => { if (!editorVisible || disabled) world.stopVoice(); }, [editorVisible, disabled, world.stopVoice]);
  useEffect(() => { if (editorVisible && !inputOpen) document.getElementById('editor-thought')?.focus({ preventScroll: true }); }, [editorVisible, inputOpen]);

  useEffect(() => {
    if (!editorVisible || help) return;
    let timer: ReturnType<typeof setTimeout>;
    const refocus = (event: Event) => {
      const target = event.target;
      if (target instanceof HTMLElement && (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable)) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (document.hasFocus()) document.getElementById(inputOpen ? 'overlay-thought' : 'editor-thought')?.focus({ preventScroll: true });
      }, 0);
    };
    window.addEventListener('focus', refocus); window.addEventListener('pointerup', refocus);
    return () => { clearTimeout(timer); window.removeEventListener('focus', refocus); window.removeEventListener('pointerup', refocus); };
  }, [editorVisible, inputOpen, help]);

  useEffect(() => {
    if (!studio || !editorVisible) return;
    const down = (event: KeyboardEvent) => {
      if (event.isComposing) return;
      if (isResetShortcut(event)) { event.preventDefault(); if (!disabled && !event.repeat) world.reset(); return; }
      if (isPushToTalk(event)) { event.preventDefault(); if (!disabled && !event.repeat) void world.startVoice('ptt'); return; }
      const typing = event.target instanceof HTMLElement && (['INPUT', 'TEXTAREA'].includes(event.target.tagName) || event.target.isContentEditable);
      if (event.key === 'Escape') { setInputOpen(false); setHelp(false); setPinnedMemory(false); return; }
      if (typing || disabled) return;
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
    const up = (event: KeyboardEvent) => { if (event.key === 'Tab') setHeldMemory(false); if (event.code === 'Backquote') world.stopPushToTalk(); };
    const blur = () => { setHeldMemory(false); world.stopPushToTalk(); };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up); window.addEventListener('blur', blur);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); window.removeEventListener('blur', blur); };
  }, [studio, editorVisible, pinnedMemory, disabled, world.undo, world.reset, world.commit, world.startVoice, world.stopPushToTalk]);

  const panel = <><div className="prompt-scroll" data-lenis-prevent onMouseDown={event => { if (event.target instanceof HTMLElement && event.target.closest('button') && document.activeElement instanceof HTMLInputElement) event.preventDefault(); }}>
    <p className="eyebrow">THE STARTING POINT</p><h3>Original prompt</h3>
    <p className="original-prompt">{originalPrompt}</p>
    <div className="prompt-provenance"><span />{endpoint ? (engine.connected ? (engine.remoteState?.fake ? 'CPU fake engine · test video' : 'Live engine connected') : 'Waiting for the engine') : live ? 'External video source' : 'Procedural world simulation'}</div>
    <div className="scene-composition"><div className="composition-heading"><span>SCENE MEMORY</span><span>{String(world.activeModifiers.length).padStart(2, '0')}</span></div>
      {!world.activeModifiers.length && <p className="untouched-note">Nothing added. Everything is possible.</p>}
      {world.activeModifiers.map(modifier => <div className="editor-modifier" key={modifier.id}>
        <span>{modifier.label}</span>{live ? <div className="live-strength">
          <button disabled={disabled} aria-label={`Less ${modifier.label}`} onClick={() => world.scrub(modifier, -0.1)}>−</button>
          <span>{Math.round(modifier.intensity * 100)}%</span><button disabled={disabled} aria-label={`More ${modifier.label}`} onClick={() => world.scrub(modifier, 0.1)}>+</button>
        </div> : <input type="range" min="0" max="1.3" step="0.05" value={modifier.intensity} aria-label={`${modifier.label} intensity`}
          onChange={event => world.scrub(modifier, Number(event.target.value) - modifier.intensity)} />}
        <button disabled={disabled} aria-label={`Remove ${modifier.label}`} onClick={() => world.remove(modifier)}>×</button>
      </div>)}
      <div className="composition-actions"><button disabled={disabled} onClick={world.undo}>↶ Undo</button><button disabled={disabled} onClick={world.reset}>Reset scene</button></div>
    </div>
    <CommandInput compact commit={localCommit} partial={localPartial} disabled={disabled} />
    <p className="server-words" data-kind={words.kind} data-unmatched={words.unmatched} aria-live="polite" aria-atomic="true">
      {keywordAt < 0 ? words.text : <>{words.text.slice(0, keywordAt)}<mark>{words.text.slice(keywordAt, keywordAt + words.keyword.length)}</mark>{words.text.slice(keywordAt + words.keyword.length)}</>}
    </p>
    <p className="editor-hint">Try <button disabled={disabled} onClick={() => void localCommit('make it winter')}>winter</button>, then <button disabled={disabled} onClick={() => void localCommit('also add a river')}>a river</button>.<br/>Your next thought builds on the last.</p>
    <button className="gesture-link" onClick={() => setHelp(true)}>Keyboard & voice gestures ↗</button>
    {endpoint && <div className="editor-timing"><p className="editor-latency" title="Command sent → first frame tagged with its chunk drawn to the source canvas. Chunk proxy; semantic change and screen paint are not measured.">{engine.latency === null ? '—' : `${engine.latency} ms`} <span>TO FIRST CORRELATED FRAME</span></p>
      <p className="server-timing">{['decided', 'embedded', 'generated', 'shown'].filter(key => engine.stamps[key] !== undefined)
        .map(key => `${key} ${engine.stamps[key]} ms`).join(' · ') || 'Measured after your first command'}</p></div>}
  </div><div className="voice-dock"><VoiceAgent visible={editorVisible} listening={world.listening}
    simulateUntil={world.simulateUntil} impulse={world.impulse} reducedMotion={reducedMotion}
    register={registerAudio} toggle={() => void world.startVoice()} disabled={disabled}
    startPushToTalk={() => void world.startVoice('ptt')} stopPushToTalk={world.stopPushToTalk} /></div></>;

  return <><ThresholdShell reducedMotion={reducedMotion} onEnter={enter} onVisibility={setEditorVisible} live={live} sourceLabel={sourceLabel} preview={preview} panel={panel}>
    <div className="living-viewport" onMouseDown={event => {
      if (event.target instanceof HTMLElement && event.target.closest('button') && document.activeElement instanceof HTMLInputElement) event.preventDefault();
    }} data-mode={live ? 'live' : 'simulation'}
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect();
        pointer.current.x = (event.clientX - rect.left) / rect.width; pointer.current.y = (event.clientY - rect.top) / rect.height;
      }}>
      {(!live || effectiveSource) && <OpticalBoundary onError={opticalError}><Suspense fallback={null}><WorldCanvas modifiers={world.activeModifiers} source={effectiveSource}
        impulse={world.impulse} pointer={pointer.current} reducedMotion={reducedMotion} active={editorVisible} onReady={worldReady} /></Suspense></OpticalBoundary>}
      {!ready && (!live || effectiveSource) && <div className="assembling-light" aria-live="polite">Gathering light…</div>}
      {(engine.notice || live && !effectiveSource) && <p className="engine-awaiting" role="status">{engine.notice || 'Waiting for the external video source…'}</p>}
      {studio && <>
        <div className="touch-surface" aria-label="Click the landscape to place your next change" onPointerDown={event => {
          if (event.button !== 0) return;
          const rect = event.currentTarget.getBoundingClientRect(); world.placePin({ x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height });
        }} />
        {!inputOpen && <WhisperCaption text={world.transcript} interim={world.interim} reducedMotion={reducedMotion} />}
        <LucidOverlay pin={world.pin} memory={heldMemory || hoverMemory || pinnedMemory} hoverMemory={setHoverMemory}
          modifiers={world.activeModifiers} remove={world.remove} scrub={world.scrub} inputOpen={inputOpen}
          closeInput={() => setInputOpen(false)} commit={localCommit} partial={localPartial} disabled={disabled} help={help} closeHelp={() => setHelp(false)}
          live={live} connected={endpoint ? engine.connected : Boolean(effectiveSource)} listening={world.listening} latency={engine.latency} undo={world.undo} reset={world.reset} />
        <AnimatePresence>{notice && <motion.p className="optical-notice" role="status" key={notice}
          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>{notice}</motion.p>}</AnimatePresence>
        <div className="sr-only" aria-live="polite">{world.activeModifiers.length ? `Active changes: ${world.activeModifiers.map(m => m.label).join(', ')}` : 'An untouched landscape.'}</div>
      </>}
    </div>
  </ThresholdShell>
  </>;
}
