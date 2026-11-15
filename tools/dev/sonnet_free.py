import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/madhacks"))
from engine.foundry import settings_from_env, FoundryMessages
s = settings_from_env(os.environ)
c = FoundryMessages.from_settings(s, timeout_s=20)
SYSTEM = """You write prompts for a text-to-video diffusion model (Wan 2.1) that is generating a live video.
You get the CURRENT prompt and the user's spoken COMMAND. Return only the complete NEW prompt, one paragraph, 60 to 120 words.
Rules: describe the whole scene every time (subjects, how many, where each is, what each stands or drives on, which way each moving thing travels relative to the camera, the camera shot and its motion, setting, light, style). Keep everything the command does not change. The model cannot follow negation: turn 'no X' into a positive description. If the command changes the camera, rewrite the camera sentence. Moving things must travel in a direction consistent with the camera motion unless the user says otherwise."""
CAR = "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast along a coastal road, moving forward away from the camera. The camera follows directly behind the car in a smooth, low tracking shot at the same speed, keeping the car centred in the frame while the white dashed centre line rushes toward the viewer. A bright turquoise ocean on the left, palm trees on the right. Sunny tropical day. Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
ARID = "a wide shot of an arid meadow with bare, dry cracked earth and a single bare oak tree with no leaves; a lion standing in the foreground"
MEADOW = "a wide shot of a meadow with a single oak tree, green grass, blue sky with white clouds, soft daylight, cinematic, realistic, highly detailed"
cases = [(ARID, "i now want a city in the night with a car driving on the road"),
         (ARID, "lots of grass"),
         (CAR, "add an airplane in the sky"),
         (CAR, "change the point of view to face the ocean, i want to see the car's side profile"),
         (CAR, "zoom out"),
         (MEADOW, "i want no clouds"),
         (MEADOW, "add another tree")]
for cur, cmd in cases:
    t = time.time()
    out = c.complete(SYSTEM, [{"role": "user", "content": f"CURRENT: {cur}\nCOMMAND: {cmd}"}])
    print(f"\n### {cmd}  ({time.time()-t:.1f} s)\n{out.strip()}")
