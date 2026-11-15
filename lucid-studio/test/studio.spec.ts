import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';

async function enter(page: Page) {
  await page.getByRole('button', { name: 'Enter the studio' }).click();
  await expect(page.locator('.experience')).toHaveClass(/editor-visible/);
  await expect(page.locator('.editor-heading')).toBeInViewport();
  await expect(page.locator('#editor-thought')).toBeFocused();
}
async function engine(page: Page) {
  const sent: Record<string, unknown>[] = [];
  let down = false;
  let control: WebSocketRoute;
  let frames: WebSocketRoute;
  await page.routeWebSocket('**/control', ws => { if (down) { ws.close(); return; } control = ws; ws.onMessage(raw => sent.push(JSON.parse(String(raw)))); ws.send(JSON.stringify({ changes: [] })); });
  await page.routeWebSocket('**/frames', ws => { frames = ws; });
  await page.goto('/?server=localhost:8100');
  await enter(page);
  await expect.poll(() => Boolean(frames)).toBe(true);
  return { sent, state: (state: object) => control.send(JSON.stringify(state)), close: () => { down = true; control.close(); }, frame: async (chunk = 1) => {
    const jpeg = await page.evaluate(() => { const c = document.createElement('canvas'); c.width = 672; c.height = 384; c.getContext('2d')!.fillRect(0, 0, 672, 384); return c.toDataURL('image/jpeg').split(',')[1]; });
    const header = Buffer.alloc(6); header.writeUInt32BE(chunk);
    frames.send(Buffer.concat([header, Buffer.from(jpeg, 'base64')]));
  } };
}
async function fakeSpeech(page: Page) {
  await page.addInitScript(() => {
    const sessions: unknown[] = [];
    const Recognition = class {
      onresult?: (e: unknown) => void; onend?: () => void; onerror?: (e: unknown) => void;
      start() { sessions.push(this); }
      stop() { this.onend?.(); }
      abort() { this.onend?.(); }
    };
    Object.assign(window, { speechSessions: sessions, webkitSpeechRecognition: Recognition, SpeechRecognition: Recognition });
  });
}
async function speech(page: Page, text: string, final = false) {
  await page.evaluate(({ text, final }) => {
    const w = window as unknown as { speechSessions: { onresult: (e: unknown) => void }[] };
    w.speechSessions.at(-1)!.onresult({ resultIndex: 0, results: [{ 0: { transcript: text }, isFinal: final }] });
  }, { text, final });
}

test('hero button scrolls naturally to a contained editor, with room for the voice dock', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('/');
  await enter(page);
  await page.locator('#editor-thought').fill('make it winter');
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toBeVisible();
  await expect(page.locator('.agent-particles canvas')).toBeVisible();
  await expect(page.locator('.hero-preview')).toHaveCSS('background-image', /data:image\/jpeg/);
  expect(errors).toEqual([]);
  for (const size of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 390, height: 844 }, { width: 375, height: 667 }]) {
    await page.setViewportSize(size);
    const boxes = await page.evaluate(() => {
      const box = (q: string) => { const r = document.querySelector(q)!.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, width: r.width }; };
      return { portal: box('.world-portal'), rail: box('.prompt-column'), dock: box('.voice-dock'), input: box('.editor-command'), scroll: box('.prompt-scroll'), overflow: document.documentElement.scrollWidth > innerWidth };
    });
    expect(boxes.overflow).toBe(false);
    expect(boxes.portal.width).toBeLessThan(size.width);
    expect(boxes.input.bottom).toBeLessThanOrEqual(boxes.dock.top);
    expect(boxes.scroll.bottom).toBeLessThanOrEqual(boxes.dock.top);
    if (size.width > 700) expect(boxes.portal.right).toBeLessThan(boxes.rail.left);
  }
});

test('Wispr inserts, typing partials, IME, button focus and disconnected text preservation', async ({ page }) => {
  const e = await engine(page);
  const input = page.locator('#editor-thought');
  await input.pressSequentially('make', { delay: 90 });
  expect(e.sent.filter(m => m.type === 'partial').map(m => m.text)).toEqual(['m', 'ma', 'mak', 'make']);
  expect(e.sent.filter(m => m.type === 'final')).toHaveLength(0);
  await input.fill('make it winter'); // multi-character insertText: Wispr whole insert
  await expect.poll(() => e.sent.filter(m => m.type === 'final').length).toBe(1);
  await expect(input).toHaveValue('');
  await page.getByRole('button', { name: 'Reset scene', exact: true }).click();
  await expect(input).toBeFocused();
  await input.dispatchEvent('compositionstart');
  await input.evaluate(el => { (el as HTMLInputElement).value = 'make it night'; el.dispatchEvent(new InputEvent('input', { bubbles: true, data: 'night', inputType: 'insertCompositionText', isComposing: true })); });
  await page.waitForTimeout(220);
  expect(e.sent.filter(m => m.type === 'final')).toHaveLength(1);
  await input.dispatchEvent('compositionend');
  await input.press('Enter');
  await expect.poll(() => e.sent.filter(m => m.type === 'final').length).toBe(2);
  await page.locator('.touch-surface').click({ position: { x: 30, y: 30 } });
  await expect(input).toBeFocused();
  await input.evaluate(el => el.blur());
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await expect(input).toBeFocused();
  e.close();
  await expect(page.getByRole('button', { name: 'Reset scene', exact: true })).toBeDisabled();
  await input.fill('add a river');
  await input.press('Enter');
  await expect(input).toHaveValue('add a river');
  await expect(page.getByRole('status').filter({ hasText: 'Connection lost' })).toBeVisible();
});

test('server words override local words even with a repeated timestamp; chips wait for server state', async ({ page }) => {
  const e = await engine(page);
  e.state({ changes: [{ id: 'w', label: 'winter', strength: 1 }], heard: 'make it winter', keyword: 'winter', kind: 'final', t_client: 100 });
  await expect(page.locator('.server-words mark')).toHaveText('winter');
  await page.getByRole('button', { name: 'Remove winter', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toBeVisible();
  e.state({ changes: [{ id: 'w', label: 'winter' }], heard: 'please sing a song', unmatched: true, t_client: 100 });
  await expect(page.locator('.server-words')).toHaveText('please sing a song');
  await expect(page.locator('.server-words')).toHaveAttribute('data-unmatched', 'true');
  e.state({ changes: [], heard: 'go back', kind: 'final', t_client: 101, stamps: { decided: 2, embedded: 3, generated: 200, shown: 210 }, fake: true });
  await expect(page.locator('.editor-source')).toContainText('CPU FAKE ENGINE');
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toHaveCount(0);
});

test('partial echoes never claim understanding; a confirmed LLM final without a keyword can', async ({ page }) => {
  const e = await engine(page);
  const input = page.locator('#editor-thought');
  await input.pressSequentially('make it night', { delay: 40 });
  e.state({ changes: [], heard: 'make it night', keyword: 'night', t_client: 1, unmatched: false });
  await expect(page.locator('.words-status')).not.toContainText('understood');
  await expect(page.locator('.words-label')).toHaveText('Hearing…');
  await input.press('Enter');
  await expect.poll(() => e.sent.some(m => m.type === 'final')).toBe(true);
  const t = e.sent.find(m => m.type === 'final')!.t_client;
  e.state({ changes: [], heard: 'make it night', t_client: t, unmatched: false });
  await expect(page.locator('.words-status')).toHaveText('Understood');
});

test('latency is measured once and a frame stall remains visible after two seconds', async ({ page }) => {
  const e = await engine(page);
  await page.locator('#editor-thought').fill('make it winter');
  await expect.poll(() => e.sent.find(m => m.type === 'final')).toBeTruthy();
  const t = e.sent.find(m => m.type === 'final')!.t_client;
  e.state({ changes: [], t_client: t, applies_from_chunk: 5, stamps: { decided: 2, embedded: 3, shown: 210 } });
  await e.frame(4);
  await expect(page.locator('.editor-latency')).toContainText('—');
  await e.frame(5);
  await expect(page.locator('.editor-latency')).toHaveText(/\d+ ms/);
  const measured = await page.locator('.editor-latency').textContent();
  await expect(page.locator('.server-timing')).toContainText('decided 2 ms');
  e.state({ changes: [], t_client: 999, applies_from_chunk: 6, stamps: { decided: 500 } });
  await expect(page.locator('.server-timing')).toContainText('decided 2 ms');
  e.state({ changes: [], t_client: t, applies_from_chunk: 5 });
  await e.frame(6);
  await expect(page.locator('.editor-latency')).toHaveText(measured!);
  await expect(page.getByRole('status').filter({ hasText: 'Video paused' })).toBeVisible({ timeout: 5000 });
  await page.waitForTimeout(3000);
  await expect(page.getByRole('status').filter({ hasText: 'Video paused' })).toBeVisible();
});

test('backtick flushes interim speech once; Ctrl stays available to Wispr and Alt+R resets', async ({ browser }) => {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  await fakeSpeech(page);
  const e = await engine(page);
  await page.keyboard.press('Control+Backquote');
  expect(await page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(0);
  await page.keyboard.down('Backquote');
  await expect.poll(() => page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(1);
  await speech(page, 'add a river');
  await speech(page, 'add a river');
  await page.keyboard.up('Backquote');
  await expect.poll(() => e.sent.filter(m => m.type === 'final').length).toBe(1);
  expect(e.sent.filter(m => m.type === 'partial')).toHaveLength(1);
  await page.keyboard.press('Alt+r');
  await expect.poll(() => e.sent.some(m => m.type === 'tap' && m.action === 'reset')).toBe(true);
  await expect(page.locator('#editor-thought')).toHaveValue('');
  await context.close();
});

test('toggle voice restarts after a natural end and errors stop it with useful text', async ({ browser }) => {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  await fakeSpeech(page);
  const e = await engine(page);
  const toggle = page.getByRole('button', { name: 'Enable voice input', exact: true });
  await toggle.click();
  await expect.poll(() => page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(1);
  await speech(page, 'make it winter', true);
  await speech(page, 'make it winter', true);
  await expect.poll(() => e.sent.filter(m => m.type === 'final').length).toBe(1);
  await page.waitForTimeout(550);
  await page.evaluate(() => (window as unknown as {speechSessions: {onend: () => void}[]}).speechSessions.at(-1)!.onend());
  await expect.poll(() => page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(2);
  await page.evaluate(() => (window as unknown as {speechSessions: {onerror: (e: unknown) => void}[]}).speechSessions.at(-1)!.onerror({error:'not-allowed'}));
  await expect(page.getByRole('status').filter({ hasText: 'Microphone blocked' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Enable voice input', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await context.close();
});

test('push-to-talk retains final segments until release and blocks a quick second start until onend', async ({ browser }) => {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  await fakeSpeech(page);
  const e = await engine(page);
  await page.keyboard.down('Backquote');
  await expect.poll(() => page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(1);
  await speech(page, 'make it night', true);
  await page.evaluate(() => {
    const rec = (window as unknown as {speechSessions: {onresult: (e: unknown) => void; stop: () => void}[]}).speechSessions.at(-1)!;
    rec.onresult({ resultIndex: 1, results: [{0:{transcript:'make it night'},isFinal:true}, {0:{transcript:'and add a river'},isFinal:true}] });
    rec.stop = () => {}; // Chrome can deliver onend later than stop().
  });
  expect(e.sent.filter(m => m.type === 'final')).toHaveLength(0);
  await page.keyboard.up('Backquote');
  await page.keyboard.down('Backquote');
  expect(await page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(1);
  await page.evaluate(() => (window as unknown as {speechSessions: {onend: () => void}[]}).speechSessions[0].onend());
  await expect.poll(() => e.sent.filter(m => m.type === 'final').length).toBe(1);
  expect(e.sent.find(m => m.type === 'final')!.text).toBe('make it night and add a river');
  await page.keyboard.up('Backquote');
  await context.close();
});

test('a late JPEG cannot overwrite a newer decoded frame or survive disconnect cleanup', async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { releaseDecode?: () => void; holdNext?: boolean; draws: number; closes: number };
    w.draws = 0; w.closes = 0;
    const draw = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function (...args: Parameters<typeof draw>) { w.draws++; return draw.apply(this, args); };
    const decode = window.createImageBitmap;
    let first = true;
    window.createImageBitmap = (async (...args: Parameters<typeof decode>) => {
      const bitmap = await decode(...args);
      const close = bitmap.close.bind(bitmap);
      bitmap.close = () => { w.closes++; close(); };
      if (first || w.holdNext) { first = false; w.holdNext = false; await new Promise<void>(resolve => { w.releaseDecode = resolve; }); }
      return bitmap;
    }) as typeof decode;
  });
  const e = await engine(page);
  await e.frame(1);
  await expect.poll(() => page.evaluate(() => Boolean((window as unknown as {releaseDecode?: () => void}).releaseDecode))).toBe(true);
  await e.frame(2);
  await expect.poll(() => page.evaluate(() => (window as unknown as {draws: number}).draws)).toBe(1);
  await page.evaluate(() => (window as unknown as {releaseDecode: () => void}).releaseDecode());
  await expect.poll(() => page.evaluate(() => (window as unknown as {closes: number}).closes)).toBe(2);
  expect(await page.evaluate(() => (window as unknown as {draws: number}).draws)).toBe(1);
  await page.evaluate(() => { const w = window as unknown as {holdNext: boolean; releaseDecode?: () => void}; w.holdNext = true; w.releaseDecode = undefined; });
  await e.frame(3);
  await expect.poll(() => page.evaluate(() => Boolean((window as unknown as {releaseDecode?: () => void}).releaseDecode))).toBe(true);
  e.close();
  await expect(page.getByRole('button', { name: 'Reset scene', exact: true })).toBeDisabled();
  await page.evaluate(() => (window as unknown as {releaseDecode: () => void}).releaseDecode());
  await expect.poll(() => page.evaluate(() => (window as unknown as {closes: number}).closes)).toBe(3);
  expect(await page.evaluate(() => (window as unknown as {draws: number}).draws)).toBe(1);
});

test('WebGL failure leaves the editor usable and clears the loading overlay', async ({ browser }) => {
  const errors: string[] = [];
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type: string, ...args: unknown[]) {
      if (type.includes('webgl')) return null;
      return getContext.apply(this, [type, ...args] as never);
    } as typeof getContext;
  });
  await page.goto('/');
  await enter(page);
  await expect(page.locator('.optical-fallback')).toBeVisible();
  await expect(page.locator('.assembling-light')).toHaveCount(0);
  await page.locator('#editor-thought').fill('make it winter');
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toBeVisible();
  await page.waitForTimeout(1800);
  await expect(page.getByRole('button', { name: 'Reset scene', exact: true })).toBeVisible();
  expect(errors.filter(e => !e.includes('Error creating WebGL context.'))).toEqual([]);
  await context.close();
});

test('an external video source supplies the hero poster without being labeled procedural or engine-disconnected', async ({ page }) => {
  await page.goto('/test/fixtures/app.html?mode=external');
  await expect(page.locator('.hero-preview')).toHaveCSS('background-image', /data:image\/jpeg/);
  const rgb = await page.evaluate(async () => {
    const url = getComputedStyle(document.querySelector('.hero-preview')!).backgroundImage.slice(5,-2);
    const image = new Image(); image.src=url; await image.decode();
    const c=document.createElement('canvas'); c.width=1;c.height=1;
    c.getContext('2d')!.drawImage(image, image.width/2, image.height/2, 1, 1, 0, 0, 1, 1);
    return [...c.getContext('2d')!.getImageData(0,0,1,1).data];
  });
  expect(rgb[0]).toBeGreaterThan(200);
  expect(rgb[1]).toBeLessThan(30);
  expect(rgb[2]).toBeLessThan(30);
  await expect(page.locator('.editor-source')).toContainText('EXTERNAL VIDEO SOURCE');
  await expect(page.locator('.prompt-provenance')).toHaveText('External video source');
  await expect(page.locator('.original-prompt')).toHaveText('A red test feed.');
});


test('a send-only adapter retains its labeled procedural simulation', async ({ page }) => {
  await page.goto('/test/fixtures/app.html?mode=adapter');
  await enter(page);
  await expect(page.locator('.living-viewport canvas')).toBeVisible();
  await expect(page.locator('.editor-source')).toContainText('PROCEDURAL STUDY');
  await expect(page.locator('.engine-awaiting')).toHaveCount(0);
  await page.locator('#editor-thought').fill('make it winter');
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toBeVisible();
  expect(await page.evaluate(() => (window as unknown as {mutations: string[]}).mutations)).toEqual(['make it winter']);
});

test('releasing push-to-talk during permission does not cancel a newer request or leak tracks', async ({ browser }) => {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  await fakeSpeech(page);
  await page.addInitScript(() => {
    const requests: { release: () => void; stream: MediaStream }[] = [];
    Object.assign(window, { audioRequests: requests });
    const request = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      const stream = await request(constraints);
      await new Promise<void>(release => requests.push({ release, stream }));
      return stream;
    };
  });
  await engine(page);
  await page.keyboard.down('Backquote');
  await expect.poll(() => page.evaluate(() => (window as unknown as {audioRequests: unknown[]}).audioRequests.length)).toBe(1);
  await page.keyboard.up('Backquote');
  await page.keyboard.down('Backquote');
  await expect.poll(() => page.evaluate(() => (window as unknown as {audioRequests: unknown[]}).audioRequests.length)).toBe(2);
  await page.evaluate(() => (window as unknown as {audioRequests: {release: () => void}[]}).audioRequests[0].release());
  await page.waitForTimeout(50);
  await page.evaluate(() => (window as unknown as {audioRequests: {release: () => void}[]}).audioRequests[1].release());
  await expect.poll(() => page.evaluate(() => (window as unknown as {speechSessions: unknown[]}).speechSessions.length)).toBe(1);
  await page.keyboard.up('Backquote');
  await expect.poll(() => page.evaluate(() => (window as unknown as {audioRequests: {stream: MediaStream}[]}).audioRequests.every(r => r.stream.getTracks().every(t => t.readyState === 'ended')))).toBe(true);
  await context.close();
});


test('custom adapter commands keep scene changes in submission order despite pending sends', async ({ page }) => {
  await page.goto('/test/fixtures/app.html?mode=delayed');
  await enter(page);
  await page.getByRole('button', { name: 'winter', exact: true }).click();
  await page.getByRole('button', { name: 'a river', exact: true }).click();
  await page.evaluate(() => (window as unknown as {pendingMutations: {release: () => void}[]}).pendingMutations[0].release());
  await expect.poll(() => page.evaluate(() => (window as unknown as {pendingMutations: unknown[]}).pendingMutations.length)).toBe(2);
  await page.evaluate(() => (window as unknown as {pendingMutations: {release: () => void}[]}).pendingMutations[1].release());
  await expect(page.getByRole('button', { name: 'Remove winter', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Remove river', exact: true })).toBeVisible();
});

test('a second whole insert is sent once when the first asynchronous send completes', async ({ page }) => {
  await page.goto('/test/fixtures/app.html?mode=delayed');
  await enter(page);
  const input = page.locator('#editor-thought');
  await input.fill('make it winter');
  await input.fill('add a river');
  await page.evaluate(() => (window as unknown as {pendingMutations: {release: () => void}[]}).pendingMutations[0].release());
  await expect.poll(() => page.evaluate(() => (window as unknown as {pendingMutations: unknown[]}).pendingMutations.length)).toBe(2);
  await page.evaluate(() => (window as unknown as {pendingMutations: {release: () => void}[]}).pendingMutations[1].release());
  await expect(input).toHaveValue('');
  expect(await page.evaluate(() => (window as unknown as {mutations: string[]}).mutations)).toEqual(['make it winter', 'add a river']);
});

test('an external source draws its poster, then stops hidden GPU redraws', async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { gpuDraws: number }; w.gpuDraws = 0;
    const draw = WebGL2RenderingContext.prototype.drawElements;
    WebGL2RenderingContext.prototype.drawElements = function (...args: Parameters<typeof draw>) { w.gpuDraws++; return draw.apply(this, args); };
  });
  await page.goto('/test/fixtures/app.html?mode=external');
  await expect(page.locator('.hero-preview')).toHaveCSS('background-image', /data:image\/jpeg/);
  await page.waitForTimeout(350);
  const draws = await page.evaluate(() => (window as unknown as {gpuDraws: number}).gpuDraws);
  await page.waitForTimeout(350);
  expect(await page.evaluate(() => (window as unknown as {gpuDraws: number}).gpuDraws)).toBe(draws);
});
