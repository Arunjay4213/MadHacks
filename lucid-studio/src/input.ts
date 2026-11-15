import type { AudioBands } from './types';

export class TranscriptGate {
  private finals = new Set<number>();
  private hold = false;
  private partial = '';
  private pending = new Map<number, string>();
  accept(index: number, text: string, final: boolean): 'partial' | 'final' | null {
    if (this.finals.has(index) || !text.trim()) return null;
    if (final) {
      this.finals.add(index);
      if (this.hold) this.pending.set(index, text);
      else this.pending.delete(index);
      this.partial = this.interim();
      return this.hold ? 'partial' : 'final';
    }
    if (this.pending.get(index) === text) return null;
    this.pending.set(index, text);
    this.partial = this.interim();
    return 'partial';
  }
  interim() { return [...this.pending.values()].join(' '); }
  flush() { const text = this.partial; this.partial = ''; this.pending.clear(); return text || null; }
  reset(hold = false) { this.hold = hold; this.finals.clear(); this.pending.clear(); this.partial = ''; }
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

// Same input policy as Ani's web/lib.js: whole dictation inserts are immediate.
export function makeInputClassifier() {
  let times: number[] = [];
  return {
    classify(inputType: string, data: string | null, now: number): 'whole' | 'burst' | 'typing' {
      if (['insertFromPaste', 'insertFromDrop', 'insertReplacementText', 'insertFromYank'].includes(inputType)
        || inputType === 'insertText' && data !== null && data.length > 1) {
        times = [];
        return 'whole';
      }
      times.push(now);
      if (times.length > 4) times.shift();
      return times.length === 4 && times.every((t, i) => i === 0 || t - times[i - 1] <= 30) ? 'burst' : 'typing';
    },
    reset() { times = []; },
  };
}
type ShortcutEvent = Pick<KeyboardEvent, 'code' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey' | 'isComposing'>;
export const isPushToTalk = (e: ShortcutEvent) => e.code === 'Backquote' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.isComposing;
export const isResetShortcut = (e: ShortcutEvent) => e.code === 'KeyR' && e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.isComposing;

export function micErrorText(error: string) {
  if (error === 'no-speech' || error === 'aborted') return null;
  if (error === 'not-allowed' || error === 'service-not-allowed') return 'Microphone blocked: allow it for this site in Chrome. You can still type or use Wispr Flow.';
  if (error === 'network') return 'The browser speech service could not be reached. It needs Google Chrome (not Brave, Arc or Opera) and no VPN or ad blocker blocking Google. Use the text box or Wispr Flow meanwhile.';
  if (error === 'audio-capture') return 'No microphone found. Use the text box or Wispr Flow.';
  return `Microphone: ${error}`;
}

const CONNECTORS = new Set(['a', 'an', 'the', 'and', 'or', 'but', 'then', 'also', 'plus', 'with', 'on', 'in', 'into',
  'onto', 'to', 'of', 'at', 'by', 'for', 'from', 'through', 'under', 'near', 'along', 'across', 'make', 'add',
  'turn', 'remove', 'it', 'is', 'some', 'very', 'really']);
export function finalizeDelay(kind: 'typing' | 'burst', text: string) {
  const t = text.trim().toLowerCase();
  if (/[,;:\-–—]$/.test(t) || CONNECTORS.has(t.split(/\s+/).at(-1)!.replace(/[^a-z']/g, ''))) return null;
  return kind === 'burst' ? 600 : 2500;
}
