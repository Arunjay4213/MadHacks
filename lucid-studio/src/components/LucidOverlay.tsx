import { AnimatePresence, motion } from 'framer-motion';
import CommandInput from './CommandInput';
import type { ActiveModifier, SpatialCoords } from '../types';

interface Props {
  pin: SpatialCoords | null;
  memory: boolean;
  hoverMemory: (show: boolean) => void;
  modifiers: ActiveModifier[];
  remove: (modifier: ActiveModifier) => void;
  scrub: (modifier: ActiveModifier, delta: number) => void;
  inputOpen: boolean;
  closeInput: () => void;
  commit: (text: string) => Promise<boolean>;
  partial: (text: string) => void;
  disabled: boolean;
  help: boolean;
  closeHelp: () => void;
  live: boolean;
  connected: boolean;
  listening: boolean;
  latency: number | null;
  undo: () => void;
  reset: () => void;
}

export default function LucidOverlay(props: Props) {
  return <>
    <AnimatePresence>
      {props.pin && <motion.div className="spatial-pin" key="pin"
        style={{ left: `${props.pin.x * 100}%`, top: `${props.pin.y * 100}%` }}
        initial={{ opacity: 0, scale: 1.7 }} animate={{ opacity: 0.85, scale: 1 }} exit={{ opacity: 0, scale: 1.6 }}
        transition={{ type: 'spring', stiffness: 110, damping: 18 }} aria-hidden="true">
        <svg viewBox="0 0 80 80" fill="none"><circle className="pin-ripple" cx="40" cy="40" r="31"/><circle cx="40" cy="40" r="19" strokeDasharray="1 5"/>
          <path d="M40 8v11M40 61v11M8 40h11M61 40h11"/><circle cx="40" cy="40" r="2"/></svg>
        <span>HERE</span>
      </motion.div>}
      {props.inputOpen && <motion.div className="thought-input"
        initial={{ opacity: 0, y: 12, filter: 'blur(8px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} exit={{ opacity: 0, y: -10 }}>
        <CommandInput autoFocus commit={props.commit} partial={props.partial} disabled={props.disabled} />
        <p>ENTER TO SHAPE <span>ESC TO DISSOLVE</span></p>
      </motion.div>}
      {props.help && <motion.section className="lucid-help" aria-label="Mirage gestures" data-lenis-prevent
        initial={{ opacity: 0, filter: 'blur(8px)' }} animate={{ opacity: 1, filter: 'blur(0px)' }} exit={{ opacity: 0 }}>
        <p className="eyebrow">A FEW WAYS TO DREAM</p>
        <h2>Nothing to learn.<br/><em>A world to change.</em></h2>
        <p>Say “make it winter”. Then “also add a river”.<br/>Say “remove the river”. The snow stays.</p>
        <dl><div><dt>/</dt><dd>Write a thought</dd></div><div><dt>V</dt><dd>{props.listening ? 'Pause' : 'Enable'} voice</dd></div>
          <div><dt>HOLD `</dt><dd>Push to talk · release to send</dd></div>
          <div><dt>⌥ / ALT R</dt><dd>Reset scene</dd></div>
          <div><dt>1 — 5</dt><dd>Try snow, river, night, sunset, rain</dd></div><div><dt>TAB</dt><dd>Hold to peek at memory</dd></div>
          <div><dt>M</dt><dd>Keep memory open for keyboard access</dd></div><div><dt>⌘ / CTRL Z</dt><dd>Undo the last change</dd></div>
          <div><dt>0</dt><dd>Begin again</dd></div><div><dt>CLICK</dt><dd>Place the next thought in space</dd></div></dl>
        <p className="source-note">{props.live ? (props.connected ? 'Connected to engine · server state is authoritative.' : 'Live engine requested · connection unavailable.') : 'Procedural world simulation · no AI model is generating these frames.'}
          {props.latency !== null && <><br/>{props.latency} ms command-to-frame proxy · semantic change is not measured.</>}
          <br/>Spatial pins are supported in the simulator; live spatial edits need backend support.</p>
        <button className="text-action" onClick={props.closeHelp}>Return to the world <span>↗</span></button>
      </motion.section>}
    </AnimatePresence>
    <div className="memory-corner" onMouseEnter={() => props.hoverMemory(true)} onMouseLeave={() => props.hoverMemory(false)}>
      <button className="memory-dot" aria-label="Peek at scene memory. Press M to keep it open." aria-expanded={props.memory}
        onFocus={() => props.hoverMemory(true)} onClick={() => props.hoverMemory(!props.memory)}><span /></button>
      <AnimatePresence>
        {props.memory && <motion.section className="memory-threads" aria-label="Active scene changes"
          initial={{ opacity: 0, x: 10, filter: 'blur(6px)' }} animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }} exit={{ opacity: 0, x: 8 }}>
          <p className="eyebrow">HELD IN THE WORLD</p>
          {!props.modifiers.length && <p className="empty-memory">An untouched landscape.</p>}
          {props.modifiers.map(modifier => <div className="memory-thread" key={modifier.id}
            onWheel={e => { e.stopPropagation(); if (!props.disabled) props.scrub(modifier, e.deltaY < 0 ? 0.05 : -0.05); }}>
            <button className="modifier-name" disabled={props.disabled} onClick={() => props.scrub(modifier, 0.1)} title="Scroll to change intensity, click to increase">
              {modifier.label}<span className="thread-intensity">{Math.round(modifier.intensity * 100)}%</span></button>
            <button className="modifier-remove" disabled={props.disabled} aria-label={`Remove ${modifier.label}`} onClick={() => props.remove(modifier)}>×</button>
          </div>)}
          <div className="memory-actions"><button disabled={props.disabled} onClick={props.undo}>Undo</button><button disabled={props.disabled} onClick={props.reset}>Begin again</button></div>
          <p className="memory-footnote">SCROLL A WORD TO SHAPE ITS INTENSITY</p>
        </motion.section>}
      </AnimatePresence>
    </div>
  </>;
}
