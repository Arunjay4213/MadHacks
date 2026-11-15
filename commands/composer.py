"""Stable subject, world and modifier wording for the text encoder."""

import re

from .state import KINDS
from .world import (car_prompt, change_wording, compose_base, is_road_scene, road_scene_subject, road_scene_world,
                    world_with_overrides)

_ARTICLE_RX = re.compile(r"^(a|an|the)\s+(.+)$", re.IGNORECASE | re.DOTALL)


def _intense(phrase: str) -> str:
    """"intense" in front of the phrase, after its article if it has one: "a large lion" -> "an intense large lion"."""
    m = _ARTICLE_RX.match(phrase)
    if m is None:
        return 'intense ' + phrase
    article = m.group(1)
    if article.lower() in ('a', 'an'):
        article = 'An' if article[0].isupper() else 'an'
    return f'{article} intense {m.group(2)}'


def strength_phrase(phrase: str, strength: float) -> str:
    """The phrase worded for its strength (0 < strength <= 1.3): a subtle hint of, a moderate amount of, as is,
    intense. engine/car_template.py uses it too, so a chip reads the same in every prompt."""
    if strength < 0.5:
        return 'a subtle hint of ' + phrase
    if strength < 1:
        return 'a moderate amount of ' + phrase
    if strength > 1:
        return _intense(phrase)
    return phrase


def _phrases(changes, world_change=None):
    """(extras, modifiers): the phrases of the non-world changes that have an effect, strength-worded, by kind.
    A change the shown world words itself ("sunny" on the snowy mountain road) takes that world's sentence."""
    extras, modifiers = [], []
    for change in sorted(changes, key=lambda c: KINDS.index(c['kind'])):
        if change['kind'] == 'world' or change['strength'] == 0:
            continue
        phrase = strength_phrase(change_wording(world_change, change), change['strength'])
        (extras if change['kind'] == 'extra' else modifiers).append(phrase)
    return extras, modifiers


def compose(snapshot):
    base, changes = snapshot['base'], snapshot['changes']
    if is_road_scene(base):
        # A road scene is always Ishaan's car template, and on it a winter season is the snowy mountain road.
        world_change, rest = road_scene_world(base, changes)
        extras, modifiers = _phrases(rest, world_change)
        return car_prompt(world_with_overrides(world_change, rest), extras, modifiers, road_scene_subject(base))
    worlds = [c for c in changes if c['kind'] == 'world' and c['strength'] > 0]  # zero strength: no effect
    extras, modifiers = _phrases(changes, worlds[-1] if worlds else None)
    if worlds:
        world = world_with_overrides(worlds[-1], changes)
        base = compose_base(base, world).partition(world)[0].rstrip(' ;,')
        return '; '.join([base, *extras, world.rstrip('.'), *modifiers])
    return '; '.join([base, *modifiers, *extras])
