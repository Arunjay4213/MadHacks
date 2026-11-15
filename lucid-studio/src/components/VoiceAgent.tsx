import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import OpticalBoundary from './OpticalBoundary';
import { useAudioBands } from '../hooks/useAudioBands';
import type { MutationImpulse } from '../types';

const AcousticSingularity = lazy(() => import('./AcousticSingularity'));

export interface AudioControls { request: () => Promise<boolean>; stop: () => void }
interface Props {
  visible: boolean;
  listening: boolean;
  simulateUntil: number;
  impulse: MutationImpulse | null;
  reducedMotion: boolean;
  register: (controls: AudioControls) => void;
  toggle: () => void;
  startPushToTalk: () => void;
  stopPushToTalk: () => void;
  disabled: boolean;
}

export default function VoiceAgent(props: Props) {
  const audio = useAudioBands(props.simulateUntil);
  const [visible, setVisible] = useState(false);
  const register = useRef(props.register);
  register.current = props.register;
  useEffect(() => { register.current({ request: audio.request, stop: audio.stop }); }, [audio.request, audio.stop]);
  const audible = audio.bands.vocal > 0.045;
  useEffect(() => {
    if (audible) { setVisible(true); return; }
    const timer = setTimeout(() => setVisible(false), 1400);
    return () => clearTimeout(timer);
  }, [audible]);

  return <div className="voice-controls"><motion.button className="voice-agent" aria-label={props.listening ? 'Pause voice input' : 'Enable voice input'} aria-pressed={props.listening}
    disabled={props.disabled} onMouseDown={event => event.preventDefault()} onClick={props.toggle} animate={{ opacity: props.visible ? 1 : 0, y: props.visible ? 0 : 20 }}
    style={{ pointerEvents: props.visible ? 'auto' : 'none' }} tabIndex={props.visible ? 0 : -1}>
    <span className={`voice-idle-orbit ${props.listening ? 'is-listening' : ''}`} aria-hidden="true" />
    {props.visible && visible && <motion.span className="agent-particles" aria-hidden="true" initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 150, damping: 24 }}>
      <OpticalBoundary fallback={null}><Suspense fallback={null}><AcousticSingularity bands={audio.bands} impulse={props.impulse} reducedMotion={props.reducedMotion} /></Suspense></OpticalBoundary>
    </motion.span>}
    <span className="agent-label">{props.listening ? 'Listening to your world' : 'Give your world a voice'}</span>
    <span className="agent-key">{props.listening ? 'V TO PAUSE' : 'CLICK TO SPEAK · V'}</span>
  </motion.button>
    <button className="push-to-talk" disabled={props.disabled} aria-label="Hold to talk" onMouseDown={event => event.preventDefault()}
      onPointerDown={event => { if (event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); props.startPushToTalk(); }}
      onPointerUp={props.stopPushToTalk} onPointerCancel={props.stopPushToTalk} onLostPointerCapture={props.stopPushToTalk}
      onKeyDown={event => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) { event.preventDefault(); props.startPushToTalk(); } }}
      onKeyUp={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); props.stopPushToTalk(); } }}
      onBlur={props.stopPushToTalk}>Hold to talk <kbd>`</kbd></button>
  </div>;
}
