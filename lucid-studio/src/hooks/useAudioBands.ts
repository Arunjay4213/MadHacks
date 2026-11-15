import { useCallback, useEffect, useRef, useState } from 'react';
import { splitBands } from '../input';
import type { AudioBands } from '../types';

const silence: AudioBands = { low: 0, vocal: 0, high: 0 };

export function useAudioBands(simulateUntil: number) {
  const [bands, setBands] = useState<AudioBands>(silence);
  const context = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const synthetic = useRef(simulateUntil);
  const generation = useRef(0);
  synthetic.current = simulateUntil;

  const stop = useCallback(() => {
    generation.current++;
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    analyser.current = null;
    void context.current?.close();
    context.current = null;
  }, []);

  const request = useCallback(async () => {
    const current = ++generation.current;
    const media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    if (current !== generation.current) { media.getTracks().forEach(track => track.stop()); return false; }
    stream.current = media;
    const ctx = new AudioContext();
    context.current = ctx;
    await ctx.resume();
    if (current !== generation.current) return false;
    const node = ctx.createAnalyser();
    node.fftSize = 512;
    node.smoothingTimeConstant = 0.72;
    ctx.createMediaStreamSource(media).connect(node);
    analyser.current = node;
    return true;
  }, []);

  useEffect(() => {
    let frame = 0;
    let last = 0;
    const spectrum = new Uint8Array(256);
    const tick = (now: number) => {
      if (now - last > 32) {
        last = now;
        let next = silence;
        if (analyser.current && context.current) {
          analyser.current.getByteFrequencyData(spectrum);
          const measured = splitBands(spectrum, context.current.sampleRate, 512);
          next = { low: Math.min(1, measured.low * 1.8), vocal: Math.min(1, measured.vocal * 2.4), high: Math.min(1, measured.high * 3) };
        }
        if (now < synthetic.current) {
          const envelope = Math.min(1, (synthetic.current - now) / 400);
          next = {
            low: Math.max(next.low, (0.35 + Math.sin(now * 0.008) ** 2 * 0.25) * envelope),
            vocal: Math.max(next.vocal, (0.32 + Math.sin(now * 0.013) ** 2 * 0.55) * envelope),
            high: Math.max(next.high, Math.sin(now * 0.027) ** 12 * 0.8 * envelope),
          };
        }
        setBands(previous => {
          if (next === silence && previous.vocal < 0.001 && previous.low < 0.001 && previous.high < 0.001) return previous;
          return { low: previous.low * 0.6 + next.low * 0.4, vocal: previous.vocal * 0.5 + next.vocal * 0.5, high: previous.high * 0.45 + next.high * 0.55 };
        });
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(frame); stop(); };
  }, [stop]);
  return { bands, request, stop };
}
