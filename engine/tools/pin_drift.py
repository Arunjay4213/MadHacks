"""List the engine/demo_prompts.json entries whose composed prompts changed under the current code: prompt_before
(the key) or canonical (the composed scene after). With --regen (needs FOUNDRY_ENDPOINT and FOUNDRY_KEY), ask the
real Sonnet final reader again for every such entry and every new entry, in chain order, and write the file back.
A new entry is {"sentence", "prompt_before": "" (or the start scene, for a reset), "operations": [], "prompt": ""},
inserted by hand where it belongs; the entry after it is asked again too when its key changes.
Read every new prompt before committing the file.

Usage, from the repo root:  python -m engine.tools.pin_drift [--regen]"""
import copy
import datetime
import json
import sys

from commands.composer import compose
from commands.state import DEFAULT_BASE, SceneState
from engine import car_template, final_reader as fr

PATH = "engine/demo_prompts.json"


def main(argv: list[str]) -> int:
    entries = json.load(open(PATH))
    regen = "--regen" in argv
    reader = fr.load("sonnet", timeout_s=30.0, cache_path=None, demo_prompts=None) if regen else None
    start = compose(SceneState(DEFAULT_BASE).snapshot())
    scene = SceneState(DEFAULT_BASE)
    drifted = 0
    for n, e in enumerate(entries):
        if e["prompt_before"] == start:
            scene = SceneState(DEFAULT_BASE)
        before = compose(scene.snapshot())
        ops = fr.expand_operations(e["operations"], scene.snapshot()) if e["operations"] else None
        trial = copy.deepcopy(scene)
        if ops:
            trial.apply(ops[0] if len(ops) == 1 else {"op": "batch", "operations": ops})
        after = compose(trial.snapshot())
        # A new entry to pin is {"sentence", "prompt_before": "" (or the start scene, for a reset), "operations": [],
        # "prompt": ""}: it is asked in the scene the entries before it lead to.
        stale = not e["prompt"] or before != e["prompt_before"] or after != e.get("canonical", after)
        if stale:
            drifted += 1
            print(f"#{n} {e['sentence']!r}: key {'same' if before == e['prompt_before'] else 'CHANGED'}, "
                  f"after {'same' if after == e.get('canonical') else 'CHANGED'}")
            if regen:
                snap = scene.snapshot()
                res = reader(e["sentence"], snap)
                if res is None or res.get("blocked"):
                    sys.exit(f"no result for #{n} {e['sentence']!r}: {res}")
                ops = fr.expand_operations(res["operations"], snap)
                trial = copy.deepcopy(scene)
                trial.apply(ops[0] if len(ops) == 1 else {"op": "batch", "operations": ops})
                e.update({"prompt_before": before, "operations": res["operations"], "prompt": res["prompt"],
                          "canonical": compose(trial.snapshot()), "source": "claude-sonnet-5-5",
                          "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
                print(f"   NEW: {res['prompt']}")
        elif regen and car_template.vehicle_subject(trial.snapshot()["base"]) is not None:
            # As live: the reader is told the road, scene and light last given for a vehicle scene, so an entry
            # asked again after an unchanged one keeps that entry's fog or light.
            reader._given_vehicle[after] = e["prompt"]
        scene = trial
    print(f"{drifted} of {len(entries)} entries drifted")
    if regen:
        open(PATH, "w").write(json.dumps(entries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
