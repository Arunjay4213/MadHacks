import { useEffect, useRef, useState } from 'react';
import ParticleOrb from './ParticleOrb';
import { useAudioBands } from '../hooks/useAudioBands';
import type { MutationImpulse } from '../types';

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

// Pressing the orb longer than this counts as hold-to-talk; a shorter press is a click that keeps listening.
const HOLD_MS = 250;

// The voice control: a particle orb. Hold it and talk, or click it once to keep listening.
export default function VoiceAgent(props: Props) {
  const audio = useAudioBands(props.simulateUntil);
  const register = useRef(props.register);
  register.current = props.register;
  useEffect(() => { register.current({ request: audio.request, stop: audio.stop }); }, [audio.request, audio.stop]);
  const [held, setHeld] = useState(false);
  const pressedAt = useRef(0);

  const press = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0 || props.disabled) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    pressedAt.current = performance.now();
    setHeld(true);
    if (!props.listening) props.startPushToTalk();
  };
  const release = () => {
    if (!pressedAt.current) return;
    const short = performance.now() - pressedAt.current < HOLD_MS;
    pressedAt.current = 0;
    setHeld(false);
    props.stopPushToTalk();
    if (short) props.toggle(); // a quick click: keep listening until clicked again
  };

  const active = props.listening || held;
  return <div className="voice-orb-row">
    <button className="orb-button" aria-label={active ? 'Listening' : 'Hold to speak'} aria-pressed={active}
      disabled={props.disabled} onMouseDown={event => event.preventDefault()}
      onPointerDown={press} onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release}
      onKeyDown={event => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) { event.preventDefault(); setHeld(true); props.startPushToTalk(); } }}
      onKeyUp={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); setHeld(false); props.stopPushToTalk(); } }}>
      <ParticleOrb level={audio.bands.vocal} listening={active} reducedMotion={props.reducedMotion} />
    </button>
    <div className="orb-text">
      <span className="orb-title">{active ? 'Listening… speak now' : 'Hold the orb and speak'}</span>
      <span className="orb-sub">{active ? (held ? 'let go when you are done' : 'click the orb to stop') : 'or hold the ` key · click once to keep listening'}</span>
    </div>
  </div>;
}
