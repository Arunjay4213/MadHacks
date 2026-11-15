# Mirage demo script

Two chains of spoken sentences that are pinned in `engine/demo_prompts.json` (the `demo_chain` field), so each one changes the video in about 0.4 s with no Sonnet call.
Any other sentence still works, through one Sonnet call, in about 2 s.
Every pinned prompt was written once by the final reader (Sonnet), read by hand, and checked on the GPU on 3 seeds at 672x384 and 2 seeds at 832x480 (2026-11-15).

How to say it:

- Say the sentences in this order, from the start scene named.
  A pin is keyed by the scene before plus the sentence, so a step only hits its pin after the steps before it.
- "reset" (or the reset button) goes back to the meadow; "undo" (or the undo button) goes back one step and is always instant.
- The variants listed under each step hit the same pin: same picture, same delay.
- Wait for the picture to settle (about 2 s) before the next sentence.

## Chain 1: the meadow

Start: press reset (the meadow with one oak tree).

| # | Say | What the audience sees | Delay on live | Variants that also hit |
|---|---|---|---|---|
| 1 | make it winter | Deep snow, the oak goes bare. | MEADOW1 | winter, make it snow, let it snow, make it snowy, turn it into winter, now make it winter, let's make it winter, make it wintery, make it wintertime |
| 2 | add a red car | A red car parked beside the oak in the snow. | MEADOW2 | add a red sports car, now add a red car, add a car, put a red car in the scene, let's add a red car, add a red sports car please |
| 3 | make it night | Night with a full moon over the snow, the car and the tree stay. | MEADOW3 | night, make it nighttime, make it night time, turn it into night, now make it night, let's make it night, make it dark |
| 4 | add jensen huang | Jensen Huang (a man with glasses in a black leather jacket) in the foreground, the car behind him. | MEADOW4 | jensen huang, add jensen, jensen, now add jensen huang, add jensen wong, jensen wong, add jensen hwang, add jenson huang, add the nvidia ceo, bring in jensen huang |
| 5 | zoom out | The camera pulls back: Jensen, the car and the tree smaller in a wide snowy field. | MEADOW5 | now zoom out, zoom out please, zoom out a bit, pull back, pull the camera back, show me a wider shot, wide shot |
| 6 | now take me to tokyo at night with neon signs | A neon street in Tokyo at night: Jensen large on the left sidewalk, the red car parked at the right curb further down the street (parked, so nothing moves near him). | MEADOW6 | take me to tokyo at night with neon signs, now take me to tokyo at night, take me to tokyo at night, take me to tokyo, now take me to tokyo, tokyo at night with neon signs, now take me to tokyo at night with neon lights |
| 7 | make it snow | Snow on the Tokyo street under the neon. | MEADOW7 | let it snow, make it snowy, add snow, now make it snow, make it winter |
| 8 | undo (optional) | Back to Tokyo without the snow. In 2 of 5 GPU runs Jensen did not come back at once, so use undo in the car chain for the wow. | MEADOW8 | the undo button |

## Chain 2: the car

Start: press reset, then say the first sentence.
Every step keeps Ishaan's measured car shot: the car seen from directly behind, driving forward away from the camera, the camera following low behind it.
It is the only driving shot the model draws forward; side and front views drove the car backwards on live, so they are not in the script.

| # | Say | What the audience sees | Delay on live | Variants that also hit |
|---|---|---|---|---|
| 1 | new scene of a red sports car driving along a country road | A red sports car seen from behind, driving fast down a country road. | CAR1 | show me a red sports car driving along a country road, a red sports car driving along a country road, new scene of a red sports car on a country road, new scene a red sports car driving along a country road |
| 2 | take me to the snowy mountains | The same car on a snowy mountain road through snowy pines, peaks ahead. | CAR2 | now take me to the snowy mountains, take me to the mountains, drive to the snowy mountains, snowy mountains, let's go to the snowy mountains |
| 3 | now a neon city at night | A neon city street at night, wet asphalt reflecting the lights. | CAR3 | a neon city at night, now take me to a neon city at night, take me to a neon city at night, neon city at night, now a neon city, make it a neon city |
| 4 | take me to a desert highway at sunset | A straight desert highway between red mesas, the sun setting ahead. | CAR4 | now take me to a desert highway at sunset, desert highway at sunset, take me to the desert at sunset, take me to a desert highway, drive to a desert highway at sunset, let's go to a desert highway at sunset |
| 5 | take me to the beach | A beachfront road at sunset, the ocean on the left, palm trees on the right. | CAR5 | now take me to the beach, drive to the beach, let's go to the beach, take me to the beach please, go to the beach |
| 6 | make it a thunderstorm | Black storm clouds, lightning over the horizon, heavy rain and spray. | CAR6 | thunderstorm, make it stormy, add a thunderstorm, now make it a thunderstorm, make it a storm, make it rain |
| 7 | make it night | A pitch dark stormy night, lightning, the red rear lights glowing on the wet road. | CAR7 | now make it night, night, make it nighttime, turn it into night, make it night time, let's make it night, make it dark |
| 8 | undo | Back to the thunderstorm. | CAR8 | the undo button |

## Unscripted sentences

Anything else goes to Sonnet and takes about 2 s (measured on live: UNSCRIPTED).
Avoid in front of judges: side or front views of a driving car (it drove backwards), aircraft (they come head-on), more than about 4 added things at once, a change of place while a car is parked, and "slowly".
