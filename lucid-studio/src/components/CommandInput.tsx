import { useEffect, useRef, type FormEvent } from 'react';

export default function CommandInput({ commit, compact = false }: { commit: (text: string) => void; compact?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const lastKey = useRef(0);
  const flush = () => {
    clearTimeout(timer.current);
    const text = input.current?.value.trim();
    if (!text) return;
    commit(text);
    if (input.current) input.current.value = '';
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  const submit = (event: FormEvent) => { event.preventDefault(); flush(); };
  return <form className={compact ? 'editor-command' : 'thought-command'} onSubmit={submit} data-lenis-prevent>
    <label htmlFor={compact ? 'editor-thought' : 'overlay-thought'}>{compact ? 'YOUR NEXT THOUGHT' : 'A THOUGHT, IN YOUR WORDS'}</label>
    <input ref={input} id={compact ? 'editor-thought' : 'overlay-thought'} placeholder="Make it winter…" autoComplete="off" maxLength={500}
      onPaste={event => { const text = event.clipboardData.getData('text'); if (text.trim()) { event.preventDefault(); commit(text.trim()); if (input.current) input.current.value = ''; } }}
      onInput={event => {
        const native = event.nativeEvent as InputEvent;
        clearTimeout(timer.current);
        if (native.isComposing) return;
        const now = performance.now();
        const machine = now - lastKey.current < 30;
        lastKey.current = now;
        timer.current = setTimeout(flush, native.inputType === 'insertReplacementText' ? 0 : machine ? 200 : 1500);
      }} />
    <button type="submit" aria-label="Apply the typed change">↗</button>
    <span className="command-caption">SPEAK, TYPE, OR PASTE FROM WISPR</span>
  </form>;
}
