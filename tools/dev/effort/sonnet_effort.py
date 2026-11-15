"""Effort and thinking sweep for the Sonnet 5.5 prompt writer (copy of ~/sonnet_free.py, varied request body).

Usage:  python sonnet_effort.py probe            -> one call per body variant, records API errors
        python sonnet_effort.py run OUT.json     -> 25 sentences x 2 repeats x each accepted setting
Calls go one at a time (no parallel calls), so latency is not inflated by our own load.
The order is shuffled across settings so a slow minute of the API hits every setting equally.
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.expanduser("~/madhacks"))
from engine.foundry import settings_from_env, FoundryMessages

s = settings_from_env(os.environ)
c = FoundryMessages.from_settings(s, timeout_s=90)

SYSTEM = """You write prompts for a text-to-video diffusion model (Wan 2.1) that is generating a live video.
You get the CURRENT prompt and the user's spoken COMMAND. Return only the complete NEW prompt, one paragraph, 60 to 120 words.
Rules: describe the whole scene every time (subjects, how many, where each is, what each stands or drives on, which way each moving thing travels relative to the camera, the camera shot and its motion, setting, light, style). Keep everything the command does not change. The model cannot follow negation: turn 'no X' into a positive description. If the command changes the camera, rewrite the camera sentence. Moving things must travel in a direction consistent with the camera motion unless the user says otherwise."""

CAR = "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast along a coastal road, moving forward away from the camera. The camera follows directly behind the car in a smooth, low tracking shot at the same speed, keeping the car centred in the frame while the white dashed centre line rushes toward the viewer. A bright turquoise ocean on the left, palm trees on the right. Sunny tropical day. Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."
ARID = "a wide shot of an arid meadow with bare, dry cracked earth and a single bare oak tree with no leaves; a lion standing in the foreground"
MEADOW = "a wide shot of a meadow with a single oak tree, green grass, blue sky with white clouds, soft daylight, cinematic, realistic, highly detailed"
DEER = "A wide, static shot of a green meadow with a single large oak tree in the centre. Two brown deer walk slowly from left to right across the grass in the middle distance, side-on to the camera. Blue sky with a few white clouds, soft afternoon daylight. The camera is fixed on a tripod at eye level and does not move. Cinematic, realistic, highly detailed."
SPRING = "A wide, static shot of a meadow in early spring with a single large oak tree in the centre, fresh green grass and small yellow wildflowers, a clear pale blue sky, soft morning light. The camera is fixed at eye level and does not move. Cinematic, realistic, highly detailed."
CITY = "A red sports car, seen from directly behind with its rear lights facing the camera, drives fast down a wide city street at night, moving forward away from the camera. The camera follows directly behind the car in a smooth, low tracking shot at the same speed, keeping the car centred. Tall glass buildings with lit windows on both sides, neon signs reflecting on the wet asphalt. Cinematic, highly detailed, realistic, sharp focus, motion blur on the road."

CASES = [  # (id, scene name, current prompt, command)
    ("s01", "ARID", ARID, "i now want a city in the night with a car driving on the road"),
    ("s02", "ARID", ARID, "lots of grass"),
    ("s03", "CAR", CAR, "add an airplane in the sky"),
    ("s04", "CAR", CAR, "change the point of view to face the ocean, i want to see the car's side profile"),
    ("s05", "CAR", CAR, "zoom out"),
    ("s06", "MEADOW", MEADOW, "i want no clouds"),
    ("s07", "MEADOW", MEADOW, "add another tree"),
    ("s08", "MEADOW", MEADOW, "make it winter"),
    ("s09", "MEADOW", MEADOW, "make it sunset"),
    ("s10", "CAR", CAR, "make it night time and raining"),
    ("s11", "CAR", CAR, "put the car on a snowy mountain pass"),
    ("s12", "MEADOW", MEADOW, "add three horses running across the field"),
    ("s13", "DEER", DEER, "get rid of the deer"),
    ("s14", "CAR", CAR, "add a motorcycle next to the car"),
    ("s15", "CAR", CAR, "no palm trees"),
    ("s16", "MEADOW", MEADOW, "turn the camera slowly to the left"),
    ("s17", "CAR", CAR, "switch to a drone shot from high above"),
    ("s18", "DEER", DEER, "make it five deer instead"),
    ("s19", "SPRING", SPRING, "go back to the snow"),
    ("s20", "CAR", CAR, "make it more dramatic"),
    ("s21", "MEADOW", MEADOW, "put a small wooden cabin to the left of the tree"),
    ("s22", "CAR", CAR, "the car should be blue"),
    ("s23", "DEER", DEER, "make them run toward the camera"),
    ("s24", "MEADOW", MEADOW, "i don't want the tree anymore, just an open field"),
    ("s25", "CITY", CITY, "actually undo that, put the ocean back"),
]

# Every body variant to try once. The run uses the ones the API accepts among the six settings asked for.
VARIANTS = {
    "off-low":       {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
    "off-medium":    {"thinking": {"type": "between_tools"}, "output_config": {"effort": "medium"}},
    "off-high":      {"thinking": {"type": "between_tools"}, "output_config": {"effort": "high"}},
    "off-xhigh":     {"thinking": {"type": "between_tools"}, "output_config": {"effort": "xhigh"}},
    "adaptive-low":  {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
    "adaptive-medium": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
    "adaptive-high": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
    "adaptive-xhigh": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
    "adaptive-max":  {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}},
    "adaptive-low-summarized": {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}},
    "omitted-low":   {"output_config": {"effort": "low"}},
    "disabled-low":  {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
    "enabled-budget-low": {"thinking": {"type": "enabled", "budget_tokens": 1024}, "output_config": {"effort": "low"}},
    "between_tools-display": {"thinking": {"type": "between_tools", "display": "summarized"}, "output_config": {"effort": "low"}},
}
RUN_SETTINGS = ["off-low", "off-medium", "off-high", "adaptive-low", "adaptive-medium", "adaptive-high",
                "adaptive-xhigh", "adaptive-max"]  # adaptive thinks only at xhigh and up on this task (probe)
MAX_TOKENS = 4096  # room for thinking at high effort; the prompt itself is about 150 tokens


def call(variant: str, cur: str, cmd: str) -> dict:
    body = c.body(SYSTEM, [{"role": "user", "content": f"CURRENT: {cur}\nCOMMAND: {cmd}"}])
    body.pop("thinking"), body.pop("output_config")
    body.update(VARIANTS[variant])
    body["max_tokens"] = MAX_TOKENS
    t = time.time()
    try:
        r = c._client.post(c._url, json=body)
    except Exception as e:  # transport error or timeout
        return {"ok": False, "latency_s": round(time.time() - t, 3), "error": f"{type(e).__name__}: {e}"}
    dt = round(time.time() - t, 3)
    if r.status_code != 200:
        return {"ok": False, "latency_s": dt, "status": r.status_code, "error": r.text[:600]}
    reply = r.json()
    text = "".join(b["text"] for b in reply["content"] if b.get("type") == "text").strip()
    thinking = [b for b in reply["content"] if b.get("type") in ("thinking", "redacted_thinking")]
    return {"ok": True, "latency_s": dt, "text": text, "stop_reason": reply.get("stop_reason"),
            "usage": reply.get("usage"), "thinking_blocks": len(thinking),
            "thinking_summary": " ".join(b.get("thinking", "") for b in thinking)[:400]}


def probe():
    _, _, cur, cmd = CASES[2]
    out = {}
    for v in VARIANTS:
        res = call(v, cur, cmd)
        out[v] = res
        msg = f"{res['latency_s']}s out_tokens={res['usage'].get('output_tokens')}" if res["ok"] else res.get("error")
        print(f"{v:28s} ok={res['ok']} {msg}", flush=True)
    print(json.dumps(out))


def run(path: str):
    jobs = [(st, case, rep) for st in RUN_SETTINGS for case in CASES for rep in range(2)]
    random.Random(7).shuffle(jobs)
    rows = []
    for i, (st, (cid, scene, cur, cmd), rep) in enumerate(jobs):
        res = call(st, cur, cmd)
        rows.append({"setting": st, "case": cid, "scene": scene, "command": cmd, "rep": rep, **res})
        print(f"[{i+1}/{len(jobs)}] {st} {cid} rep{rep} ok={res['ok']} {res['latency_s']}s", flush=True)
        with open(path, "w") as f:
            json.dump({"system": SYSTEM, "max_tokens": MAX_TOKENS, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    probe() if sys.argv[1] == "probe" else run(sys.argv[2])
