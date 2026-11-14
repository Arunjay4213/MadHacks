import { useEffect, useRef, useState, type ReactNode } from 'react';
import { motion } from 'framer-motion';
import Lenis from 'lenis';

interface Props {
  children: ReactNode;
  reducedMotion: boolean;
  onEnter: () => void;
  onVisibility: (visible: boolean) => void;
  live: boolean;
  sourceLabel: string;
  preview?: string;
  panel: ReactNode;
}

export default function ThresholdShell({ children, reducedMotion, onEnter, onVisibility, live, sourceLabel, preview, panel }: Props) {
  const editor = useRef<HTMLElement>(null);
  const entered = useRef(false);
  const enterRef = useRef(onEnter);
  const visibilityRef = useRef(onVisibility);
  const [visible, setVisible] = useState(false);
  enterRef.current = onEnter;
  visibilityRef.current = onVisibility;
  const lenisRef = useRef<Lenis>();

  useEffect(() => {
    const lenis = new Lenis({ lerp: 0.065, smoothWheel: !reducedMotion });
    lenisRef.current = lenis;
    let frame = 0;
    const tick = (time: number) => { lenis.raf(time); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    const observer = new IntersectionObserver(([entry]) => {
      const inView = entry.isIntersecting;
      setVisible(inView);
      visibilityRef.current(inView);
      if (inView && !entered.current) { entered.current = true; enterRef.current(); }
    }, { threshold: 0.15 });
    if (editor.current) observer.observe(editor.current);
    return () => { cancelAnimationFrame(frame); lenis.destroy(); observer.disconnect(); };
  }, [reducedMotion]);

  const enter = () => { if (editor.current) lenisRef.current?.scrollTo(editor.current, { duration: reducedMotion ? 0 : 1.15 }); };

  return <div className={`experience ${visible ? 'editor-visible' : ''}`}>
    <section className="hero-section" aria-label="Lucid introduction">
      <header className="threshold-header"><span className="wordmark">lucid<span className="wordmark-period">.</span></span>
        <span className="edition">A STUDY IN LIVING WORLDS <span>№ 001</span></span></header>
      <div className="threshold-title">
        <motion.p initial={{ opacity: 0, filter: 'blur(6px)' }} animate={{ opacity: 1, filter: 'blur(0px)' }} transition={{ duration: 1.2 }} className="eyebrow">THOUGHT BECOMES LANDSCAPE</motion.p>
        <motion.h1 initial={{ opacity: 0, y: 14, filter: 'blur(8px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} transition={{ duration: 1.4, delay: 0.15 }}>
          A world, <em>within a word.</em>
        </motion.h1>
        <motion.p className="threshold-description" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7, duration: 1.2 }}>
          Speak a thought. Watch it become.
        </motion.p>
      </div>
      <button className="hero-preview" onClick={enter} aria-label="Open the Lucid editor" style={preview ? { backgroundImage: `url(${preview})` } : {}}>
        <span className="preview-atmosphere" />
        <span className="portal-caption"><span>THE WORLD IS LISTENING</span><span>{live ? 'ENGINE / PREVIEW' : 'ALPINE STUDY / PREVIEW'}</span></span>
      </button>
      <footer className="threshold-footer"><span className="entry-mode">{live ? sourceLabel : 'PROCEDURAL WORLD SIMULATION'}</span>
        <button className="entry-trigger" onClick={enter}>Enter the studio <span aria-hidden="true">↓</span></button>
        <span className="entry-footnote">YOUR VOICE. AN INFINITE CANVAS.</span></footer>
    </section>
    <section className="editor-section" ref={editor} id="studio" aria-label="Generative video editor">
      <header className="editor-heading"><div><span className="eyebrow">LUCID STUDIO</span><h2>A landscape in progress.</h2></div>
        <span className="editor-source">{sourceLabel} <i /></span></header>
      <div className="editor-grid"><div className="generation-column">
        <div className="generation-heading"><span>CURRENT GENERATION</span><span>CONTINUOUS / 001</span></div>
        <main className="world-portal" aria-label="Current video generation">{children}</main>
        <div className="canvas-footnote"><span>Speak to evolve the scene. Each thought stays until you change it.</span><span>CLICK TO PLACE A THOUGHT ↗</span></div>
      </div><aside className="prompt-column" aria-label="Prompt and scene controls">{panel}</aside></div>
      <footer className="editor-footer"><span>BUILT ON TRAINING-FREE INTERACTIVE WORLD MODELS</span><span>WORDS BECOME WORLDS.</span></footer>
    </section>
  </div>;
}
