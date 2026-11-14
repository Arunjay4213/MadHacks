// MOCK server for developing the page without the engine. It is NOT the real server.
// It loops a test pattern, keeps a tiny authoritative state, and follows the PROVISIONAL
// transport in PLAN.md (not yet agreed by the lead). Every message says fake: true.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { WebSocketServer } from 'ws';

const here = dirname(fileURLToPath(import.meta.url));
const STATIC = { '/': 'index.html', '/index.html': 'index.html', '/style.css': 'style.css', '/app.js': 'app.js', '/lib.js': 'lib.js' };
const TYPES = { html: 'text/html; charset=utf-8', css: 'text/css', js: 'text/javascript' };
const FPS = 16;
const FRAMES_PER_CHUNK = 12; // 0.75 s chunks at 16 fps, as in PLAN.md
const MAX_MSG_BYTES = 4096;

// shortcut: a dozen demo keywords, not Ani's command reader; replaced by the real server.
const KEYWORDS = { winter: 'winter', snow: 'winter', river: 'river', stream: 'river', night: 'night', rain: 'rain', fog: 'fog', sunset: 'sunset', autumn: 'autumn', storm: 'storm' };

function makeFrames() {
  const r = spawnSync('ffmpeg', ['-loglevel', 'error', '-f', 'lavfi', '-i', `testsrc=size=416x240:rate=${FPS}`,
    '-frames:v', String(FRAMES_PER_CHUNK * 4), '-f', 'image2pipe', '-c:v', 'mjpeg', '-q:v', '5', '-'], { maxBuffer: 64 << 20 });
  const frames = [];
  if (r.status === 0) {
    // JPEG entropy data stuffs 0xFF bytes, so 0xFFD9 only appears as the end-of-image marker.
    let start = 0;
    for (let i = 0; i < r.stdout.length - 1; i++) {
      if (r.stdout[i] === 0xff && r.stdout[i + 1] === 0xd9) {
        frames.push(r.stdout.subarray(start, i + 2));
        start = i + 2;
      }
    }
  }
  if (frames.length) return frames;
  // No ffmpeg: one 1x1 grey JPEG, so the page and tests still run.
  return [Buffer.from('/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==', 'base64')];
}

export function createState() {
  let changes = [];
  const history = [];
  let nextId = 1;
  const commit = (next) => {
    history.push(changes);
    changes = next;
  };
  return {
    get changes() {
      return changes;
    },
    // Returns {keyword, unmatched} for a final sentence.
    applyText(text) {
      const t = text.toLowerCase();
      if (/\b(undo|go back)\b/.test(t)) return { keyword: /undo/.test(t) ? 'undo' : 'go back', unmatched: !this.undo() };
      const kw = Object.keys(KEYWORDS).find((k) => new RegExp(`\\b${k}`).test(t));
      if (!kw) return { unmatched: true };
      const label = KEYWORDS[kw];
      if (/\b(remove|no more|without)\b/.test(t)) {
        const hit = changes.findLast((c) => c.label === label);
        return { keyword: kw, unmatched: !(hit && this.remove(hit.id)) };
      }
      if (!changes.some((c) => c.label === label)) commit([...changes, { id: `c${nextId++}`, label, strength: 1 }]);
      return { keyword: kw, unmatched: false };
    },
    spot(text) {
      const t = text.toLowerCase();
      return Object.keys(KEYWORDS).find((k) => t.includes(k));
    },
    remove(id) {
      if (!changes.some((c) => c.id === id)) return false;
      commit(changes.filter((c) => c.id !== id));
      return true;
    },
    undo() {
      if (!history.length) return false;
      changes = history.pop();
      return true;
    },
    reset() {
      if (changes.length) commit([]);
    },
  };
}

export async function startServer({ port = 8787, host = '127.0.0.1' } = {}) {
  const frames = makeFrames();
  const state = createState();
  let frameIndex = 0;
  const chunk = () => Math.floor(frameIndex / FRAMES_PER_CHUNK);

  const http = createServer(async (req, res) => {
    const name = STATIC[new URL(req.url, 'http://x').pathname];
    if (!name) return res.writeHead(404).end('not found');
    try {
      const body = await readFile(join(here, name));
      res.writeHead(200, { 'content-type': TYPES[name.split('.').pop()], 'cache-control': 'no-store' }).end(body);
    } catch {
      res.writeHead(404).end('not found');
    }
  });

  const control = new WebSocketServer({ noServer: true, maxPayload: MAX_MSG_BYTES });
  const video = new WebSocketServer({ noServer: true, maxPayload: MAX_MSG_BYTES });
  http.on('upgrade', (req, socket, head) => {
    const path = new URL(req.url, 'http://x').pathname;
    const wss = path === '/control' ? control : path === '/frames' ? video : null;
    if (!wss) return socket.destroy();
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  });

  const snapshot = (extra = {}) => JSON.stringify({ fake: true, chunk: chunk(), changes: state.changes, ...extra });
  const broadcast = (s) => control.clients.forEach((c) => c.readyState === 1 && c.send(s));

  control.on('connection', (ws) => {
    ws.send(snapshot());
    ws.on('message', (data, isBinary) => {
      const t0 = performance.now();
      let m;
      try {
        m = isBinary ? null : JSON.parse(data.toString());
      } catch {}
      if (!m || typeof m !== 'object') return ws.send(JSON.stringify({ fake: true, error: 'bad message: not JSON' }));
      const echo = Number.isFinite(m.t_client) ? { t_client: m.t_client } : {};
      if ((m.type === 'partial' || m.type === 'final') && typeof m.text === 'string') {
        if (m.type === 'partial') return broadcast(snapshot({ kind: 'partial', heard: m.text, keyword: state.spot(m.text) }));
        const r = state.applyText(m.text);
        const decided = Math.round(performance.now() - t0);
        // Mock stamps: only what this process actually measured.
        return broadcast(snapshot({ kind: 'final', heard: m.text, ...r, ...echo, applies_from_chunk: chunk() + 1, stamps: { heard: 0, decided } }));
      }
      if (m.type === 'tap') {
        if (m.action === 'remove' && typeof m.id === 'string') state.remove(m.id);
        else if (m.action === 'undo') state.undo();
        else if (m.action === 'reset') state.reset();
        else return ws.send(JSON.stringify({ fake: true, error: `bad tap action` }));
        return broadcast(snapshot({ applies_from_chunk: chunk() + 1 }));
      }
      ws.send(JSON.stringify({ fake: true, error: 'bad message: unknown type' }));
    });
  });

  const timer = setInterval(() => {
    const header = Buffer.alloc(4);
    header.writeUInt32BE(chunk());
    const msg = Buffer.concat([header, frames[frameIndex % frames.length]]);
    video.clients.forEach((c) => c.readyState === 1 && c.send(msg));
    frameIndex++;
  }, 1000 / FPS);

  await new Promise((resolve) => http.listen(port, host, resolve));
  return {
    port: http.address().port,
    close: () =>
      new Promise((resolve) => {
        clearInterval(timer);
        [...control.clients, ...video.clients].forEach((c) => c.terminate());
        http.close(resolve);
      }),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const s = await startServer({ port: Number(process.env.PORT) || 8787 });
  console.log(`MOCK fake server on http://127.0.0.1:${s.port} (not the real engine)`);
}
