import test from 'node:test';
import assert from 'node:assert/strict';
import WebSocket from 'ws';
import { startServer, createState } from '../fake-server.mjs';
import { parseServerMessage, parseFrame } from '../lib.js';

test('state: add, stack, remove by name, undo, reset', () => {
  const s = createState();
  assert.equal(s.applyText('make it winter').unmatched, false);
  s.applyText('also add a river');
  s.applyText('slowly turn to night');
  assert.deepEqual(s.changes.map((c) => c.label), ['winter', 'river', 'night']);
  assert.equal(s.applyText('remove the river').unmatched, false);
  assert.deepEqual(s.changes.map((c) => c.label), ['winter', 'night']);
  s.applyText('go back');
  assert.deepEqual(s.changes.map((c) => c.label), ['winter', 'river', 'night']);
  assert.equal(s.applyText('turn the tree into a lighthouse').unmatched, true);
  s.reset();
  assert.deepEqual(s.changes, []);
  s.undo();
  assert.equal(s.changes.length, 3);
});

function client(port, path) {
  const ws = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  const queue = [];
  const waiters = [];
  ws.on('message', (d, bin) => {
    const v = bin ? d : d.toString();
    const w = waiters.shift();
    w ? w(v) : queue.push(v);
  });
  return {
    ws,
    open: () => new Promise((r) => ws.once('open', r)),
    next: () => (queue.length ? Promise.resolve(queue.shift()) : new Promise((r) => waiters.push(r))),
  };
}

test('server: control round trip is authoritative, and bad input never crashes it', async (t) => {
  const srv = await startServer({ port: 0 });
  t.after(() => srv.close());
  const c = client(srv.port, '/control');
  await c.open();
  const next = async () => parseServerMessage(await c.next());

  const hello = await next();
  assert.equal(hello.fake, true);
  assert.deepEqual(hello.changes, []);

  c.ws.send(JSON.stringify({ type: 'partial', text: 'make it win', t_client: 1 }));
  const p = await next();
  assert.equal(p.kind, 'partial');
  assert.deepEqual(p.changes, []); // partial never changes state

  c.ws.send(JSON.stringify({ type: 'final', text: 'make it winter', t_client: 123 }));
  const f = await next();
  assert.equal(f.t_client, 123);
  assert.equal(f.keyword, 'winter');
  assert.ok(Number.isInteger(f.applies_from_chunk));
  assert.deepEqual(f.changes.map((x) => x.label), ['winter']);

  c.ws.send('not json');
  assert.match((await next()).error, /bad message/);
  c.ws.send(JSON.stringify({ type: 'tap', action: 'explode' }));
  assert.match((await next()).error, /bad tap/);

  c.ws.send(JSON.stringify({ type: 'tap', action: 'remove', id: f.changes[0].id }));
  assert.deepEqual((await next()).changes, []);
  c.ws.send(JSON.stringify({ type: 'tap', action: 'undo' }));
  assert.deepEqual((await next()).changes.map((x) => x.label), ['winter']);
  c.ws.send(JSON.stringify({ type: 'tap', action: 'reset' }));
  assert.deepEqual((await next()).changes, []);
  c.ws.close();
});

test('server: frames are tagged with a chunk number and carry a JPEG', async (t) => {
  const srv = await startServer({ port: 0 });
  t.after(() => srv.close());
  const c = client(srv.port, '/frames');
  await c.open();
  const d = await c.next();
  const f = parseFrame(d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength));
  assert.ok(Number.isInteger(f.chunk));
  assert.deepEqual([...f.jpeg.subarray(0, 2)], [0xff, 0xd8]);
  c.ws.close();
});

test('server: serves only whitelisted files', async (t) => {
  const srv = await startServer({ port: 0 });
  t.after(() => srv.close());
  const base = `http://127.0.0.1:${srv.port}`;
  assert.equal((await fetch(`${base}/lib.js`)).status, 200);
  assert.equal((await fetch(`${base}/package.json`)).status, 404);
  assert.equal((await fetch(`${base}/..%2fAGENTS.md`)).status, 404);
});
