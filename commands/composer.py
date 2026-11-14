"""Stable subject, world and modifier wording for the text encoder."""

from .state import KINDS
from .world import compose_base, world_with_overrides


def compose(snapshot):
    worlds = [c for c in snapshot['changes'] if c['kind'] == 'world']
    base, world, style = snapshot['base'], None, ''
    if worlds:
        world = world_with_overrides(worlds[-1], snapshot['changes'])
        rendered = compose_base(base, world)
        base, _, style = rendered.partition(world)
        base = base.rstrip(' ;,')
    extras, modifiers = [], []
    for change in sorted(snapshot['changes'], key=lambda c: KINDS.index(c['kind'])):
        if change['kind'] == 'world':
            continue
        strength = change['strength']
        if strength == 0:
            continue
        phrase = change['phrase']
        if strength < 0.5:
            phrase = 'a subtle hint of ' + phrase
        elif strength < 1:
            phrase = 'a moderate amount of ' + phrase
        elif strength > 1:
            phrase = 'intense ' + phrase
        (extras if change['kind'] == 'extra' else modifiers).append(phrase)
    if world:
        return '; '.join([base, *extras, world.rstrip('.'), *modifiers, *([style.strip()] if style.strip() else [])])
    return '; '.join([base, *modifiers, *extras])
