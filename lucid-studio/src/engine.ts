import { useEffect, useRef, useState } from 'react';
import type { ActiveModifier, WorldStreamAdapter } from './types';

export interface EngineState {
  modifiers?: ActiveModifier[];
  unmatched: boolean;
  heard: string | null;
  keyword: string;
  kind: 'partial' | 'final' | 'server';
  fake: boolean;
  confirmed?: boolean;
  stamps: Record<string, number>;
  tClient: number | null;
  appliesFromChunk: number | null;
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const chunk = (v: unknown): v is number => finite(v) && Number.isInteger(v) && v >= 0;
export function parseEngineState(value: unknown): EngineState | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const m = value as Record<string, unknown>;
  let modifiers: ActiveModifier[] | undefined;
  if ('changes' in m) {
    if (!Array.isArray(m.changes) || !m.changes.every(c => c && typeof c.id === 'string' && typeof c.label === 'string')) return null;
    modifiers = m.changes.map(c => ({ id: c.id, label: c.label, intensity: finite(c.strength) ? c.strength : 1 }));
  }
  return {
    modifiers, unmatched: m.unmatched === true,
    heard: typeof m.heard === 'string' ? m.heard : null,
    keyword: typeof m.keyword === 'string' ? m.keyword : '',
    kind: m.kind === 'partial' || m.kind === 'final' ? m.kind : 'server',
    fake: m.fake === true,
    stamps: m.stamps && typeof m.stamps === 'object' && !Array.isArray(m.stamps)
      ? Object.fromEntries(Object.entries(m.stamps).filter(([, v]) => finite(v))) : {},
    tClient: finite(m.t_client) ? m.t_client : null,
    appliesFromChunk: chunk(m.applies_from_chunk) ? m.applies_from_chunk : null,
  };
}
export function parseFrame(buffer: ArrayBuffer) {
  if (buffer.byteLength < 10) return null;
  const bytes = new Uint8Array(buffer);
  if (bytes[6] !== 255 || bytes[7] !== 216) return null;
  const view = new DataView(buffer);
  return { chunk: view.getUint32(0), index: view.getUint16(4), jpeg: buffer.slice(6) };
}

export function engineBaseUrl(server: string | null, location: { protocol: string; host: string }) {
  const secure = location.protocol === 'https:';
  const address = server || `${secure ? 'wss' : 'ws'}://${location.host}`;
  const base = new URL(address.includes('://') ? address : `${secure ? 'wss' : 'ws'}://${address}`);
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(base.protocol) || base.username || base.password) throw new Error('Invalid engine URL');
  base.protocol = ['https:', 'wss:'].includes(base.protocol) ? 'wss:' : 'ws:';
  if (secure && base.protocol === 'ws:') throw new Error('HTTPS requires a secure engine URL');
  return base.origin;
}
interface Link { controlOpen: boolean; framesOpen: boolean; lastFrameAt: number | null; everConnected: boolean }
export function videoNotice(link: Link, now: number) {
  if (!link.controlOpen || !link.framesOpen) return link.everConnected ? 'Connection lost, reconnecting…' : 'Connecting to the video server…';
  if (link.lastFrameAt === null) return 'Waiting for the first frame…';
  return now - link.lastFrameAt > 2000 ? 'Video paused: no frames from the server' : null;
}
export function makeLatencyTracker() {
  const sent = new Set<number>();
  const draws: { chunk: number; at: number }[] = [];
  let pending: { t: number; chunk: number; stamps: Record<string, number> } | null = null;
  const measure = (at: number) => {
    if (!pending) return null;
    const sample = { ms: Math.round(at - pending.t), stamps: { ...pending.stamps } };
    sent.delete(pending.t); pending = null;
    return sample;
  };
  return {
    sent(t: number) { sent.add(t); if (sent.size > 100) sent.delete(sent.values().next().value!); },
    onServer(m: Pick<EngineState, 'tClient' | 'appliesFromChunk' | 'unmatched'> & { stamps?: Record<string, number> }) {
      if (m.tClient === null || m.appliesFromChunk === null || !sent.has(m.tClient) || m.unmatched) return null;
      if (pending?.t === m.tClient) pending.stamps = m.stamps ?? {};
      else pending = { t: m.tClient, chunk: m.appliesFromChunk, stamps: m.stamps ?? {} };
      const drawn = draws.find(d => d.chunk >= pending!.chunk && d.at >= pending!.t);
      return drawn ? measure(drawn.at) : null;
    },
    onFrameDrawn(chunk: number, now = Date.now()) {
      draws.push({ chunk, at: now });
      if (draws.length > 100) draws.shift();
      return pending && chunk >= pending.chunk && now >= pending.t ? measure(now) : null;
    },
    reset() { sent.clear(); draws.length = 0; pending = null; },
  };
}

// Mirrors web/lib.js: an echoed draft is not a confirmed operation.
export function makeHeardTracker() {
  const partials = new Map<string, number>();
  const finals: { text: string; t: number }[] = [];
  return {
    sentPartial(text: string, now = Date.now()) {
      partials.set(text, now);
      if (partials.size > 50) partials.delete(partials.keys().next().value!);
    },
    sentFinal(text: string, t: number) { partials.clear(); finals.push({ text, t }); if (finals.length > 50) finals.shift(); },
    clearPartials() { partials.clear(); },
    classify(text: string | null, t: number | null, now = Date.now()) {
      if (finals.some(f => f.text === text && f.t === t)) return 'confirmed';
      const sentAt = text === null ? undefined : partials.get(text);
      return sentAt !== undefined && now - sentAt < 5000 ? 'pending' : 'other';
    },
  };
}

export function useEngineStream(endpoint: string | null, notify: (text: string) => void) {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [state, setState] = useState<EngineState | null>(null);
  const [connected, setConnected] = useState(false);
  const [sample, setSample] = useState<{ ms: number; stamps: Record<string, number> } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const control = useRef<WebSocket | null>(null);
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const timing = useRef(makeLatencyTracker());
  const heard = useRef(makeHeardTracker());

  useEffect(() => {
    setCanvas(null); setState(null); setConnected(false); setSample(null);
    if (!endpoint) { setNotice(null); return; }
    let active = true;
    let retry: ReturnType<typeof setTimeout>;
    let attempts = 0;
    let frames: WebSocket | null = null;
    let frameGeneration = 0;
    const link: Link = { controlOpen: false, framesOpen: false, lastFrameAt: null, everConnected: false };
    const renderNotice = () => setNotice(videoNotice(link, performance.now()));
    const target = document.createElement('canvas');
    const context = target.getContext('2d');
    let base: string;
    try { base = engineBaseUrl(endpoint, window.location); }
    catch {
      setNotice('Engine address unavailable. Use a host:port or HTTP(S)/WS(S) address; HTTPS requires WSS.');
      return;
    }
    renderNotice();
    const poll = setInterval(renderNotice, 250);
    const open = () => {
      if (!active) return;
      const ws = new WebSocket(base + '/control');
      control.current = ws;
      const epoch = ++frameGeneration;
      let received = 0;
      let drawn = 0;
      ws.onopen = () => {
        if (!active) return ws.close();
        attempts = 0;
        link.controlOpen = true;
        setConnected(true);
        frames = new WebSocket(base + '/frames');
        frames.binaryType = 'arraybuffer';
        frames.onopen = () => { if (!active || epoch !== frameGeneration) return; link.framesOpen = true; link.everConnected = true; renderNotice(); };
        frames.onmessage = async event => {
          if (!(event.data instanceof ArrayBuffer)) return;
          const frame = parseFrame(event.data);
          if (!frame || !context) return;
          const sequence = ++received;
          try {
            const bitmap = await createImageBitmap(new Blob([frame.jpeg], { type: 'image/jpeg' }));
            if (active && epoch === frameGeneration && sequence > drawn) {
              drawn = sequence;
              if (target.width !== bitmap.width || target.height !== bitmap.height) {
                target.width = bitmap.width; target.height = bitmap.height;
              }
              context.drawImage(bitmap, 0, 0);
              target.dataset.frameVersion = `${epoch}-${sequence}`;
              setCanvas(previous => previous ?? target);
              link.lastFrameAt = performance.now();
              const measured = timing.current.onFrameDrawn(frame.chunk);
              if (measured !== null) setSample(measured);
              renderNotice();
            }
            bitmap.close();
          } catch { /* Keep the last good frame; the stall notice covers repeated decode failures. */ }
        };
        frames.onclose = () => {
          if (!active || epoch !== frameGeneration) return;
          frameGeneration++;
          link.framesOpen = false; link.lastFrameAt = null;
          renderNotice();
          if (ws.readyState === WebSocket.OPEN) ws.close();
        };
      };
      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (data && typeof data.error === 'string') { notifyRef.current(data.error); return; }
          const incoming = parseEngineState(data);
          if (!incoming) { notifyRef.current('Ignored an invalid engine message'); return; }
          const status = heard.current.classify(incoming.heard, incoming.tClient);
          incoming.confirmed = status === 'confirmed' && incoming.kind !== 'partial';
          if (status === 'pending') { incoming.kind = 'partial'; incoming.keyword = ''; incoming.unmatched = false; }
          const measured = timing.current.onServer(incoming);
          if (measured !== null) setSample(measured);
          setState(previous => {
            const next = { ...incoming, modifiers: incoming.modifiers ?? previous?.modifiers };
            return JSON.stringify(previous) === JSON.stringify(next) ? previous : next;
          });
        } catch { notifyRef.current('An engine message could not be read'); }
      };
      ws.onclose = () => {
        frameGeneration++;
        frames?.close();
        if (!active) return;
        link.controlOpen = false; link.framesOpen = false; link.lastFrameAt = null;
        setConnected(false); setSample(null); timing.current.reset();
        renderNotice();
        retry = setTimeout(open, Math.min(5000, 250 * 2 ** attempts++));
      };
    };
    open();
    return () => {
      active = false; frameGeneration++;
      clearTimeout(retry); clearInterval(poll);
      control.current?.close(); frames?.close(); control.current = null;
      timing.current.reset();
    };
  }, [endpoint]);

  const sendText = (type: 'partial' | 'final', text: string) => {
    if (control.current?.readyState !== WebSocket.OPEN) throw new Error('Engine disconnected. Command was not queued.');
    const timestamp = Date.now();
    if (type === 'final') { setSample(null); timing.current.sent(timestamp); heard.current.sentFinal(text, timestamp); }
    else heard.current.sentPartial(text);
    control.current.send(JSON.stringify({ type, text, t_client: timestamp }));
  };
  const adapter: WorldStreamAdapter = { sendMutation: async payload => sendText('final', payload.prompt) };
  const tap = (action: 'remove' | 'undo' | 'reset', id?: string) => {
    if (control.current?.readyState !== WebSocket.OPEN) throw new Error('Engine disconnected. Command was not queued.');
    control.current.send(JSON.stringify({ type: 'tap', action, ...(id ? { id } : {}) }));
  };
  return { source: canvas, remoteState: state, connected, latency: sample?.ms ?? null, stamps: sample?.stamps ?? {}, clearPartials: () => heard.current.clearPartials(), notice, adapter, sendText, tap };
}
