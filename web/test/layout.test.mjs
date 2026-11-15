import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Wispr Flow pastes into #command, so the box must stay focusable in every layout, full screen
// included: it may be hidden visually, but never with display: none or visibility: hidden.
const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, selector, body]) => ({ selector: selector.trim(), body }));

test('the command box and its wrapper are never display:none or visibility:hidden', () => {
  const boxRules = rules.filter((r) => /(^|[\s,>])(\.type|#command|#command-form)(?![\w-])/.test(r.selector));
  assert.ok(boxRules.length > 0, 'no rules found for the command box');
  for (const r of boxRules) {
    assert.doesNotMatch(r.body, /display\s*:\s*none/, `${r.selector} hides the command box with display: none`);
    assert.doesNotMatch(r.body, /visibility\s*:\s*hidden/, `${r.selector} hides the command box with visibility: hidden`);
  }
});
