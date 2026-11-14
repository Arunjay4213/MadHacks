import { useCallback, useEffect, useRef, useState } from 'react';
import { speechConstructor, TranscriptGate } from '../input';
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
  const desired = useRef(false);
  const recognition = useRef<SpeechRecognizer | null>(null);
  const restart = useRef<ReturnType<typeof setTimeout>>();
  const dissolve = useRef<ReturnType<typeof setTimeout>>();
  const pinTimer = useRef<ReturnType<typeof setTimeout>>();
  const gate = useRef(new TranscriptGate());
  const partialText = useRef(false);
  const pinRef = useRef(pin);
  pinRef.current = pin;
  const activeModifiers = options.remoteModifiers ?? world.modifiers;

  const showWords = useCallback((text: string, partial = false) => {
    clearTimeout(dissolve.current);
    setTranscript(text);
    setInterim(partial);
    partialText.current = partial;
    if (!partial) dissolve.current = setTimeout(() => setTranscript(''), 1100);
  }, []);

  const commit = useCallback(async (prompt: string, simulate = true) => {
    const text = prompt.trim();
    if (!text) return;
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
        return;
      }
    }
    if (opts.current.remoteModifiers === undefined) {
      if (!intent) { opts.current.notify('Try “make it winter”, “add a river”, or “turn to night”.'); return; }
      worldRef.current = next;
      setWorld(next);
    }
    setImpulse({ id: Date.now(), startedAt: performance.now(), localized: intent?.kind === 'remove' || intent?.kind === 'add' && intent.label === 'river', center: coords ?? undefined });
    setPin(null);
  }, [showWords]);

  const stopVoice = useCallback(() => {
    desired.current = false;
    clearTimeout(restart.current);
    recognition.current?.abort();
    recognition.current = null;
    if (partialText.current) { setTranscript(''); setInterim(false); partialText.current = false; }
    setListening(false);
    opts.current.stopAudio();
  }, []);

  const startVoice = useCallback(async () => {
    if (desired.current) { stopVoice(); return; }
    const Recognition = speechConstructor();
    if (!Recognition || !navigator.mediaDevices?.getUserMedia) {
      opts.current.notify('Voice is unavailable here. Press / to type, or 1–5 to try a change.');
      return;
    }
    desired.current = true;
    try {
      if (!await opts.current.requestAudio() || !desired.current) return;
      const begin = () => {
        if (!desired.current) return;
        const rec = new Recognition();
        recognition.current = rec;
        gate.current.reset();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-US';
        rec.onresult = event => {
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const text = result[0].transcript.trim();
            const accepted = gate.current.accept(i, text, result.isFinal);
            if (accepted === 'final') void commit(text, false);
            if (accepted === 'partial') {
              showWords(text, true);
              try { opts.current.partial?.(text); } catch { /* reconnect notice is owned by the stream hook */ }
            }
          }
        };
        rec.onerror = event => {
          if (event.error === 'no-speech' || event.error === 'aborted') return;
          opts.current.notify(event.error === 'not-allowed'
            ? 'Microphone blocked. Press / to type; 1–5 simulate voice.'
            : 'Speech service unavailable. Press / to type; V tries voice again.');
          stopVoice();
        };
        rec.onend = () => {
          if (partialText.current) { setTranscript(''); setInterim(false); partialText.current = false; }
          if (desired.current) restart.current = setTimeout(begin, 250);
        };
        try { rec.start(); setListening(true); }
        catch { opts.current.notify('Voice could not start. Press / to type.'); stopVoice(); }
      };
      begin();
    } catch {
      opts.current.notify('Microphone unavailable. Press / to type; 1–5 simulate voice.');
      stopVoice();
    }
  }, [commit, showWords, stopVoice]);

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
    stopVoice(); clearTimeout(dissolve.current); clearTimeout(pinTimer.current);
  }, [stopVoice]);

  return { activeModifiers, transcript, interim, listening, impulse, simulateUntil, pin,
    commit, startVoice, stopVoice, placePin, remove, scrub, undo, reset };
}
