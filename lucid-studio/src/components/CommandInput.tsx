import { useEffect, useRef, type FormEvent } from 'react';
import { makeInputClassifier, finalizeDelay } from '../input';

interface Props {
  commit: (text: string) => Promise<boolean>;
  partial: (text: string) => void;
  compact?: boolean;
  disabled?: boolean;
  autoFocus?: boolean;
}
export default function CommandInput({ commit, partial, compact = false, disabled = false, autoFocus = false }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const classifier = useRef(makeInputClassifier());
  const composing = useRef(false);
  const sending = useRef(false);
  const pending = useRef<number | null>(null);
  const revision = useRef(0);
  const mounted = useRef(true);
  const blocked = useRef(disabled);
  blocked.current = disabled;
  const flush = async () => {
    clearTimeout(timer.current);
    if (blocked.current || composing.current || !mounted.current) return;
    if (sending.current) { pending.current = revision.current; return; }
    const version = revision.current;
    const value = input.current?.value;
    const text = value?.trim();
    if (!text) return;
    classifier.current.reset();
    sending.current = true;
    try {
      if (await commit(text) && input.current && input.current.value === value && revision.current === version) input.current.value = '';
    } finally {
      sending.current = false;
      const retry = pending.current; pending.current = null;
      if (retry !== null && retry === revision.current) void flush();
    }
  };
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; clearTimeout(timer.current); }; }, []);
  useEffect(() => { if (disabled) clearTimeout(timer.current); return () => clearTimeout(timer.current); }, [disabled]);
  const submit = (event: FormEvent) => { event.preventDefault(); void flush(); };
  const change = (inputType: string, data: string | null) => {
    revision.current++; pending.current = null;
    clearTimeout(timer.current);
    if (composing.current || disabled) return;
    const kind = classifier.current.classify(inputType, data, performance.now());
    if (kind === 'whole') { void flush(); return; }
    const text = input.current?.value.trim();
    if (!text) { partial(''); return; }
    partial(text);
    const delay = finalizeDelay(kind, text);
    if (delay !== null) timer.current = setTimeout(() => void flush(), delay);
  };
  return <form className={compact ? 'editor-command' : 'thought-command'} onSubmit={submit} data-lenis-prevent>
    <label htmlFor={compact ? 'editor-thought' : 'overlay-thought'}>{compact ? 'YOUR NEXT THOUGHT' : 'A THOUGHT, IN YOUR WORDS'}</label>
    <input ref={input} id={compact ? 'editor-thought' : 'overlay-thought'} placeholder="Make it winter… ↵" autoComplete="off" maxLength={500} autoFocus={autoFocus}
      onCompositionStart={() => { revision.current++; pending.current = null; composing.current = true; clearTimeout(timer.current); }}
      onCompositionEnd={() => { composing.current = false; change('insertCompositionText', null); }}
      onKeyDown={event => { if (event.key === 'Enter' && (event.nativeEvent.isComposing || composing.current)) event.preventDefault(); }}
      onInput={event => {
        const native = event.nativeEvent as InputEvent;
        if (native.isComposing) { clearTimeout(timer.current); return; }
        change(native.inputType, native.data);
      }} />
    <button type="submit" disabled={disabled} onMouseDown={event => event.preventDefault()} aria-label="Apply the typed change">↗</button>
    <span className="command-caption">{disabled ? 'RECONNECTING · YOUR TEXT STAYS HERE' : 'SPEAK, TYPE, OR PASTE FROM WISPR'}</span>
  </form>;
}
