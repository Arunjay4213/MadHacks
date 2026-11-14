import { useCallback, useEffect, useRef, useState } from 'react';
import { speechConstructor, TranscriptGate, micErrorText } from '../input';
import type { SpeechRecognizer } from '../input';
import { emptyWorld, evolve, parseIntent } from '../world';
import type { ActiveModifier, MutationImpulse, SpatialCoords, WorldStreamAdapter } from '../types';

interface Options {
  adapter?: WorldStreamAdapter;
  remoteModifiers?: ActiveModifier[];
  partial?: (text: string) => void;
  tap?: (action: 'remove' | 'undo' | 'reset', id?: string) => void;
  notify: (message: string) => void;
  requestAudio: () => Promise<boolean>;
  stopAudio: () => void;
}
export function useGenerativeWorld(options: Options) {
  const [world, setWorld] = useState(emptyWorld);
  const worldRef = useRef(world);
  const [transcript, setTranscript] = useState('');
  const [interim, setInterim] = useState(false);
  const [listening, setListening] = useState(false);
  const [impulse, setImpulse] = useState<MutationImpulse | null>(null);
  const [simulateUntil, setSimulateUntil] = useState(0);
  const [pin, setPin] = useState<SpatialCoords | null>(null);
  const opts = useRef(options);
  opts.current = options;
  const desired = useRef<'toggle' | 'ptt' | null>(null);
  const voiceGeneration = useRef(0);
  const recognition = useRef<SpeechRecognizer | null>(null);
  const restart = useRef<ReturnType<typeof setTimeout>>();
  const dissolve = useRef<ReturnType<typeof setTimeout>>();
  const pinTimer = useRef<ReturnType<typeof setTimeout>>();
  const gate = useRef(new TranscriptGate());
  const pinRef = useRef(pin);
  pinRef.current = pin;
  const activeModifiers = options.remoteModifiers ?? world.modifiers;

  const showWords = useCallback((text: string, partial = false) => {
    clearTimeout(dissolve.current);
    setTranscript(text);
    setInterim(partial);
    if (!partial) dissolve.current = setTimeout(() => setTranscript(''), 1100);
  }, []);

  const commit = useCallback(async (prompt: string, simulate = true) => {
    const text = prompt.trim();
    if (!text) return false;
    showWords(text);
    if (simulate) setSimulateUntil(performance.now() + 1500);
    const intent = parseIntent(text, opts.current.remoteModifiers ?? worldRef.current.modifiers);
    const coords = pinRef.current;
    const spatialIntent = intent?.kind === 'add' && coords ? { ...intent, coords } : intent;
    const next = spatialIntent ? evolve(worldRef.current, spatialIntent) : worldRef.current;
    if (opts.current.adapter) {
      try {
        await opts.current.adapter.sendMutation({ prompt: text,
          activeState: (opts.current.remoteModifiers ?? next.modifiers).map(m => m.label),
          spatialCoords: coords, timestamp: Date.now() });
      } catch (error) {
        opts.current.notify(error instanceof Error ? error.message : 'The command could not be sent');
        return false;
      }
    }
    if (opts.current.remoteModifiers === undefined) {
      if (!intent) { opts.current.notify('Try “make it winter”, “add a river”, or “turn to night”.'); return false; }
      worldRef.current = next;
      setWorld(next);
    }
    setImpulse({ id: Date.now(), startedAt: performance.now(), localized: intent?.kind === 'remove' || intent?.kind === 'add' && intent.label === 'river', center: coords ?? undefined });
    setPin(null);
    return true;
  }, [showWords]);

  const partial = useCallback((text: string) => {
    showWords(text, true);
    try { opts.current.partial?.(text); }
    catch (error) { opts.current.notify(error instanceof Error ? error.message : 'The partial could not be sent'); }
  }, [showWords]);

  const stopVoice = useCallback(() => {
    voiceGeneration.current++;
    desired.current = null;
    clearTimeout(restart.current);
    recognition.current?.stop(); // onend flushes the last interim after any final result.
    setListening(false);
    opts.current.stopAudio();
  }, []);

  const startVoice = useCallback(async (mode: 'toggle' | 'ptt' = 'toggle') => {
    if (desired.current) {
      if (mode === 'toggle' && desired.current === 'toggle') stopVoice();
      return;
    }
    if (recognition.current) return;
    const Recognition = speechConstructor();
    if (!Recognition || !navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      opts.current.notify('Voice needs Chrome on HTTPS or localhost. Use the text box or Wispr Flow.');
      return;
    }
    desired.current = mode;
    const generation = ++voiceGeneration.current;
    try {
      const available = await opts.current.requestAudio();
      if (generation !== voiceGeneration.current) return;
      if (!available || !desired.current) { stopVoice(); return; }
      const begin = () => {
        if (!desired.current) return;
        const rec = new Recognition();
        recognition.current = rec;
        gate.current.reset(desired.current === 'ptt');
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-US';
        const started = performance.now();
        let lastInterim = '';
        rec.onresult = event => {
          if (recognition.current !== rec) return;
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const text = result[0].transcript.trim();
            const accepted = gate.current.accept(i, text, result.isFinal);
            if (accepted === 'final') void commit(text, false);
          }
          const interim = gate.current.interim();
          if (interim && interim !== lastInterim) partial(interim);
          lastInterim = interim;
        };
        rec.onerror = event => {
          const message = micErrorText(event.error);
          if (message) { opts.current.notify(message); stopVoice(); }
        };
        rec.onend = () => {
          if (recognition.current !== rec) return;
          recognition.current = null;
          const leftover = gate.current.flush();
          if (leftover) void commit(leftover, false);
          if (desired.current === 'toggle' && performance.now() - started > 500) restart.current = setTimeout(begin, 250);
          else stopVoice();
        };
        try { rec.start(); setListening(true); }
        catch { recognition.current = null; opts.current.notify('Voice could not start. Use the text box.'); stopVoice(); }
      };
      begin();
    } catch {
      if (generation !== voiceGeneration.current) return;
      opts.current.notify('Microphone unavailable. Allow it for this site, or use the text box.');
      stopVoice();
    }
  }, [commit, partial, stopVoice]);

  const stopPushToTalk = useCallback(() => { if (desired.current === 'ptt') stopVoice(); }, [stopVoice]);

  const placePin = useCallback((coords: SpatialCoords) => {
    setPin(coords);
    clearTimeout(pinTimer.current);
    pinTimer.current = setTimeout(() => setPin(null), 4000);
  }, []);

  const remove = useCallback((modifier: ActiveModifier) => {
    if (!opts.current.tap) { void commit(`remove ${modifier.label}`); return; }
    try { opts.current.tap('remove', modifier.id); showWords(`remove ${modifier.label}`); }
    catch (error) { opts.current.notify(String(error)); }
  }, [commit, showWords]);

  const scrub = useCallback((modifier: ActiveModifier, delta: number) => {
    if (opts.current.adapter) {
      void commit(`${delta > 0 ? 'more' : 'less'} ${modifier.label}`);
      return;
    }
    const next = evolve(worldRef.current, { kind: 'intensity', id: modifier.id, intensity: modifier.intensity + delta });
    worldRef.current = next;
    setWorld(next);
    setSimulateUntil(performance.now() + 450);
  }, [commit]);

  const undo = useCallback(() => {
    if (!opts.current.tap) { void commit('undo'); return; }
    try { opts.current.tap('undo'); showWords('go back'); } catch (error) { opts.current.notify(String(error)); }
  }, [commit, showWords]);
  const reset = useCallback(() => {
    if (!opts.current.tap) { void commit('clear everything'); return; }
    try { opts.current.tap('reset'); showWords('clear everything'); } catch (error) { opts.current.notify(String(error)); }
  }, [commit, showWords]);

  useEffect(() => () => {
    desired.current = null;
    voiceGeneration.current++;
    clearTimeout(restart.current);
    const rec = recognition.current;
    if (rec) { rec.onend = null; rec.onresult = null; rec.onerror = null; rec.abort(); }
    recognition.current = null;
    opts.current.stopAudio();
    clearTimeout(dissolve.current); clearTimeout(pinTimer.current);
  }, [stopVoice]);

  return { activeModifiers, transcript, interim, listening, impulse, simulateUntil, pin,
    commit, partial, startVoice, stopVoice, stopPushToTalk, placePin, remove, scrub, undo, reset };
}
