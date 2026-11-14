import type { AudioBands } from './types';

export class TranscriptGate {
  private finals = new Set<number>();
  private partial = '';
  accept(index: number, text: string, final: boolean): 'partial' | 'final' | null {
    if (this.finals.has(index) || !text.trim()) return null;
    if (final) {
      this.finals.add(index);
      this.partial = '';
      return 'final';
    }
    if (this.partial === text) return null;
    this.partial = text;
    return 'partial';
  }
  reset() { this.finals.clear(); this.partial = ''; }
}

export function splitBands(data: Uint8Array, sampleRate: number, fftSize: number): AudioBands {
  const band = (from: number, to: number) => {
    const start = Math.max(0, Math.ceil(from * fftSize / sampleRate));
    const end = Math.min(data.length - 1, Math.floor(to * fftSize / sampleRate));
    let sum = 0;
    for (let i = start; i <= end; i++) sum += data[i];
    return end >= start ? sum / ((end - start + 1) * 255) : 0;
  };
  return { low: band(80, 250), vocal: band(300, 3000), high: band(4000, 10000) };
}

export interface SpeechResult { isFinal: boolean; [index: number]: { transcript: string } }
export interface SpeechEvent { resultIndex: number; results: ArrayLike<SpeechResult> }
export interface SpeechRecognizer {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechEvent) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
export function speechConstructor(): (new () => SpeechRecognizer) | undefined {
  const browser = window as unknown as { SpeechRecognition?: new () => SpeechRecognizer; webkitSpeechRecognition?: new () => SpeechRecognizer };
  return browser.SpeechRecognition ?? browser.webkitSpeechRecognition;
}
