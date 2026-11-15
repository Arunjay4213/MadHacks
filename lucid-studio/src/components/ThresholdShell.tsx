import { useEffect, useRef, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  reducedMotion: boolean;
  onEnter: () => void;
  onVisibility: (visible: boolean) => void;
  live: boolean;
  sourceLabel: string;
  preview?: string;
  panel: ReactNode;
  command: ReactNode;
  headerExtra?: ReactNode;
}

// Demo layout: the Mirage logo, the video, and one command bar. No intro screen, no scrolling, no side panel.
export default function ThresholdShell(props: Props) {
  const { children, onEnter, onVisibility, command, headerExtra } = props;
  const enterRef = useRef(onEnter);
  const visibilityRef = useRef(onVisibility);
  enterRef.current = onEnter;
  visibilityRef.current = onVisibility;

  useEffect(() => { visibilityRef.current(true); enterRef.current(); }, []);

  return <div className="experience editor-visible studio-only">
    <section className="editor-section studio-min" id="studio" aria-label="Mirage">
      <header className="studio-bar"><span className="wordmark">mirage<span className="wordmark-period">.</span></span>{headerExtra}</header>
      <main className="world-portal" aria-label="Current video generation">{children}</main>
      <div className="studio-command">{command}</div>
    </section>
  </div>;
}
