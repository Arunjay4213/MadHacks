"""Stable subject, world and modifier wording for the text encoder."""

from .state import KINDS
from .world import car_prompt, compose_base, is_road_scene, road_scene_world, world_with_overrides


def _phrases(changes):
    """(extras, modifiers): the phrases of the non-world changes that have an effect, strength-worded, by kind."""
    extras, modifiers = [], []
    for change in sorted(changes, key=lambda c: KINDS.index(c['kind'])):
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
    return extras, modifiers


def compose(snapshot):
    base, changes = snapshot['base'], snapshot['changes']
    if is_road_scene(base):
        # A road scene is always Ishaan's car template, and on it a winter season is the snowy mountain road.
        world_change, rest = road_scene_world(base, changes)
        extras, modifiers = _phrases(rest)
        return car_prompt(world_with_overrides(world_change, rest), extras, modifiers)
    worlds = [c for c in changes if c['kind'] == 'world' and c['strength'] > 0]  # zero strength: no effect
    extras, modifiers = _phrases(changes)
    if worlds:
        world = world_with_overrides(worlds[-1], changes)
        base = compose_base(base, world).partition(world)[0].rstrip(' ;,')
        return '; '.join([base, *extras, world.rstrip('.'), *modifiers])
    return '; '.join([base, *modifiers, *extras])
