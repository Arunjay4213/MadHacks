import test from 'node:test';
import assert from 'node:assert/strict';
import { makeTypeBox, typeBoxAttrs } from '../lib.js';

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

test('type box: typing reveals a hidden box; dictation and whole inserts (Wispr Flow) do not', () => {
  const { tb } = box();
  assert.equal(tb.afterInput('typing'), 'reveal-soon');
  assert.equal(tb.afterInput('burst'), 'cancel-reveal');
  assert.equal(tb.afterInput('whole'), 'cancel-reveal');
  tb.show();
  assert.equal(tb.afterInput('typing'), null); // open stays open, including after a send
});
