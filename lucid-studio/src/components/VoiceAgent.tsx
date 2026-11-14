import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
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

  return <motion.button className="voice-agent" aria-label={props.listening ? 'Pause voice input' : 'Enable voice input'} aria-pressed={props.listening}
    onClick={props.toggle} animate={{ opacity: props.visible ? 1 : 0, y: props.visible ? 0 : 20 }}
    style={{ pointerEvents: props.visible ? 'auto' : 'none' }} tabIndex={props.visible ? 0 : -1}>
    <span className={`voice-idle-orbit ${props.listening ? 'is-listening' : ''}`} aria-hidden="true" />
    {props.visible && visible && <motion.span className="agent-particles" aria-hidden="true" initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 150, damping: 24 }}>
      <Suspense fallback={null}><AcousticSingularity bands={audio.bands} impulse={props.impulse} reducedMotion={props.reducedMotion} /></Suspense>
    </motion.span>}
    <span className="agent-label">{props.listening ? 'Listening to your world' : 'Give your world a voice'}</span>
    <span className="agent-key">{props.listening ? 'V TO PAUSE' : 'CLICK TO SPEAK · V'}</span>
  </motion.button>;
}
