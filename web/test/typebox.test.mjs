import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { makeTypeBox, typeBoxAttrs, TYPE_REVEAL_MS } from '../lib.js';

function box() {
  const seen = [];
  return { tb: makeTypeBox((attrs) => seen.push(attrs)), seen };
}

test('type box: the Keyboard button toggles aria-expanded (and data-open)', () => {
  const { tb, seen } = box();
  assert.equal(tb.open, false);
  tb.toggle();
  assert.deepEqual(seen.at(-1), { 'aria-expanded': 'true', 'data-open': 'true' });
  tb.toggle();
  assert.deepEqual(seen.at(-1), { 'aria-expanded': 'false', 'data-open': 'false' });
  assert.equal(seen.length, 2);
  tb.hide(); // already hidden: nothing to render
  assert.equal(seen.length, 2);
  assert.deepEqual(typeBoxAttrs(false), { 'aria-expanded': 'false', 'data-open': 'false' });
});

test('type box: Esc hides it only while it is open and empty', () => {
  const { tb } = box();
  assert.equal(tb.closesOnKey('Escape', ''), false); // hidden already
  tb.show();
  assert.equal(tb.closesOnKey('Escape', 'make it'), false); // keeps a draft
  assert.equal(tb.closesOnKey('Escape', '  '), true);
  assert.equal(tb.closesOnKey('Enter', ''), false);
});

test('type box: typing reveals a hidden box once it pauses; a 30 ms dictation burst and a paste never do', () => {
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    const { tb } = box();
    // Wispr Flow-style burst: keys 30 ms apart; makeInputClassifier calls the 4th key a burst.
    for (const kind of ['typing', 'typing', 'typing', 'burst']) {
      tb.input(kind);
      mock.timers.tick(30);
    }
    mock.timers.tick(500);
    assert.equal(tb.open, false, 'a dictated burst opened the box');
    tb.input('whole'); // a paste
    mock.timers.tick(500);
    assert.equal(tb.open, false);
    // a person typing: the box opens once the keys pause for TYPE_REVEAL_MS
    tb.input('typing');
    mock.timers.tick(TYPE_REVEAL_MS - 1);
    assert.equal(tb.open, false);
    mock.timers.tick(1);
    assert.equal(tb.open, true);
    tb.input('typing'); // open stays open, including after a send
    assert.equal(tb.open, true);
  } finally {
    mock.timers.reset();
  }
});
