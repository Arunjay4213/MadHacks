import { useEffect, useRef, useState } from 'react';
import type { ActiveModifier, WorldStreamAdapter } from './types';

export interface EngineState {
  modifiers: ActiveModifier[];
  unmatched: boolean;
  heard: string | null;
  tClient: number | null;
  appliesFromChunk: number | null;
}
export function parseEngineState(value: unknown): EngineState | null {
  if (!value || typeof value !== 'object') return null;
  const m = value as Record<string, unknown>;
  if (!Array.isArray(m.changes) || !m.changes.every(c => c && typeof c.id === 'string'
    && typeof c.label === 'string' && typeof c.strength === 'number' && Number.isFinite(c.strength))) return null;
  return {
    modifiers: m.changes.map(c => ({ id: c.id, label: c.label, intensity: c.strength })),
    unmatched: m.unmatched === true,
    heard: typeof m.heard === 'string' ? m.heard : null,
    tClient: typeof m.t_client === 'number' ? m.t_client : null,
    appliesFromChunk: typeof m.applies_from_chunk === 'number' ? m.applies_from_chunk : null,
  };
}
export function parseFrame(buffer: ArrayBuffer) {
  if (buffer.byteLength < 10) return null;
  const bytes = new Uint8Array(buffer);
  if (bytes[6] !== 255 || bytes[7] !== 216) return null;
  const view = new DataView(buffer);
  return { chunk: view.getUint32(0), index: view.getUint16(4), jpeg: buffer.slice(6) };
}

export function useEngineStream(endpoint: string | null, notify: (text: string) => void) {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [state, setState] = useState<EngineState | null>(null);
  const [connected, setConnected] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const control = useRef<WebSocket | null>(null);
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const sent = useRef(new Map<number, number>());

  useEffect(() => {
    if (!endpoint) return;
    let active = true;
    let retry: ReturnType<typeof setTimeout>;
    let attempts = 0;
    let frames: WebSocket | null = null;
    let latest: EngineState | null = null;
    let lastFrame = -1;
    let frameGeneration = 0;
    const target = document.createElement('canvas');
    const context = target.getContext('2d');
    let base: URL;
    try {
      base = new URL(endpoint);
      if (!['http:', 'https:', 'ws:', 'wss:'].includes(base.protocol)) throw new Error();
      base.protocol = ['https:', 'wss:'].includes(base.protocol) ? 'wss:' : 'ws:';
      if (window.location.protocol === 'https:' && base.protocol === 'ws:') throw new Error();
    } catch {
      notifyRef.current('Engine address unavailable. Use an http(s) or ws(s) address; HTTPS requires WSS.');
      return;
    }
    const url = (path: string) => new URL(path, base).toString();
    const open = () => {
      if (!active) return;
      const ws = new WebSocket(url('/control'));
      control.current = ws;
      ws.onopen = () => {
        if (!active) return ws.close();
        attempts = 0;
        setConnected(true);
        const epoch = ++frameGeneration;
        lastFrame = -1;
        notifyRef.current('Live engine connected');
        frames = new WebSocket(url('/frames'));
        frames.binaryType = 'arraybuffer';
        frames.onmessage = async event => {
          if (!(event.data instanceof ArrayBuffer)) return;
          const frame = parseFrame(event.data);
          if (!frame || !context) return;
          const sequence = frame.chunk * 65536 + frame.index;
          try {
            const bitmap = await createImageBitmap(new Blob([frame.jpeg], { type: 'image/jpeg' }));
            if (active && epoch === frameGeneration && sequence > lastFrame) {
              lastFrame = sequence;
              if (target.width !== bitmap.width || target.height !== bitmap.height) {
                target.width = bitmap.width; target.height = bitmap.height;
              }
              context.drawImage(bitmap, 0, 0);
              target.dataset.frameVersion = String(sequence);
              setCanvas(previous => previous ?? target);
              const started = latest?.tClient != null ? sent.current.get(latest.tClient) : undefined;
              if (started !== undefined && latest?.appliesFromChunk != null && frame.chunk >= latest.appliesFromChunk) {
                setLatency(Math.round(performance.now() - started));
                sent.current.delete(latest.tClient!);
              }
            }
            bitmap.close();
          } catch { notifyRef.current('A video frame could not be decoded'); }
        };
        frames.onclose = () => { if (active && ws.readyState === WebSocket.OPEN) ws.close(); };
      };
      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (typeof data.error === 'string') { notifyRef.current(data.error); return; }
          const incoming = parseEngineState(data);
          if (incoming) {
            latest = incoming;
            setState(previous => JSON.stringify(previous) === JSON.stringify(incoming) ? previous : incoming);
            if (incoming.unmatched) notifyRef.current('That change was not understood. Try a simpler description.');
          }
        } catch { notifyRef.current('An engine message could not be read'); }
      };
      ws.onclose = () => {
        frames?.close();
        if (!active) return;
        setConnected(false);
        frameGeneration++;
        sent.current.clear();
        notifyRef.current('Connection lost. Reconnecting; unsent commands are discarded.');
        retry = setTimeout(open, Math.min(5000, 500 * 2 ** attempts++));
      };
    };
    open();
    return () => {
      active = false;
      clearTimeout(retry);
      control.current?.close();
      frames?.close();
      control.current = null;
      sent.current.clear();
    };
  }, [endpoint]);

  const sendText = (type: 'partial' | 'final', text: string, spatialCoords?: { x: number; y: number } | null) => {
    if (control.current?.readyState !== WebSocket.OPEN) throw new Error('Engine disconnected. Command was not queued.');
    const timestamp = Date.now();
    if (type === 'final') setLatency(null);
    sent.current.set(timestamp, performance.now());
    if (sent.current.size > 100) sent.current.delete(sent.current.keys().next().value!);
    control.current.send(JSON.stringify({ type, text, t_client: timestamp, ...(spatialCoords ? { spatialCoords } : {}) }));
  };
  const adapter: WorldStreamAdapter = {
    sendMutation: async payload => sendText('final', payload.prompt, payload.spatialCoords),
  };
  const tap = (action: 'remove' | 'undo' | 'reset', id?: string) => {
    if (control.current?.readyState !== WebSocket.OPEN) throw new Error('Engine disconnected. Command was not queued.');
    control.current.send(JSON.stringify({ type: 'tap', action, ...(id ? { id } : {}), t_client: Date.now() }));
  };
  return { source: canvas, remoteState: state, connected, latency, adapter, sendText, tap };
}
