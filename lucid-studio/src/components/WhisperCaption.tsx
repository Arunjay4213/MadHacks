import { AnimatePresence, motion } from 'framer-motion';

export default function WhisperCaption({ text, interim, reducedMotion }: { text: string; interim: boolean; reducedMotion: boolean }) {
  return <div className="whisper-position" aria-live="polite" aria-atomic="true">
    <AnimatePresence mode="wait">
      {text && <motion.div key="caption" className={`whisper-caption ${interim ? 'is-interim' : ''}`}
        exit={{ opacity: 0, y: reducedMotion ? 0 : -14, filter: 'blur(8px)' }} transition={{ duration: 0.65 }}>
        {text.split(/\s+/).map((word, index) => <motion.span key={`${index}-${word}`}
          initial={{ opacity: 0, filter: reducedMotion ? 'none' : 'blur(8px)', y: reducedMotion ? 0 : 6 }}
          animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
          transition={{ type: 'spring', stiffness: 450, damping: 32, delay: reducedMotion ? 0 : index * 0.018 }}>
          {word}{' '}
        </motion.span>)}
      </motion.div>}
    </AnimatePresence>
  </div>;
}
