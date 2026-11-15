"""Does the car drive forward? An optical-flow check on the road in front of the camera, over a run of frames.

    python -m engine.tools.motion_check DIR [DIR ...]     # each DIR holds the frames of one step, in name order

On the car template (seen from behind, the camera following low) the ground beside the car (road edges, lane lines,
verges) moves down the frame, toward the viewer, while the car drives forward. A car that reverses, stops or turns the
shot moves it up, not at all, or sideways. The car itself sits in the middle of the frame and moves with the camera,
and the asphalt under it has little texture, so the check looks at the lower left and lower right of the frame
(SIDES boxes). For each pair of frames 1/8 s apart it takes the mean vertical flow (Farneback, in pixels) of the
pixels that move at all (MOVING px or more), over both boxes. A step is FORWARD when the median over its pairs is
above MIN_DY, REVERSE when it is below -MIN_DY, and UNCLEAR in between (a snowy or dark scene has little texture
beside the road). It is a heuristic: read it with the frames.
The first SKIP_S seconds after a command are skipped (the dissolve to the new scene), and pairs across a scene cut
(a mean absolute frame difference over CUT_DIFF, as at a refresh) are left out.
Needs OpenCV (the engine machines have it in ~/engine-venv); prints one JSON line per DIR.
"""

import enum
import json
import sys
from pathlib import Path

SIDES = ((0.55, 1.0, 0.0, 0.28), (0.55, 1.0, 0.72, 1.0))  # top, bottom, left, right, as shares of the frame
MOVING = 1.0                     # pixels of flow for a pixel to count as moving
MIN_DY = 0.3                     # pixels of median downward flow
STRIDE = 2                       # frames between the two of a pair (1/8 s at 16 fps)
SKIP_S = 1.0
FPS = 16
CUT_DIFF = 40.0                  # mean absolute grey difference of two frames: a cut, not motion


class Motion(enum.Enum):
    FORWARD = "forward"          # the ground beside the road moves down the frame, toward the viewer
    REVERSE = "reverse"          # it moves up: the car drives backwards
    UNCLEAR = "unclear"          # too little motion to tell
    TOO_FEW = "too few frames"


def road_flow(prev, cur) -> float | None:
    """Mean vertical flow (pixels, + is down) of the moving pixels in the SIDES boxes between two BGR frames; None
    across a cut."""
    import cv2
    import numpy as np
    a, b = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur))
    if float(np.mean(cv2.absdiff(a, b))) > CUT_DIFF:
        return None
    h, w = a.shape
    dys = []
    for top, bottom, left, right in SIDES:
        t, bt, l, r = int(top * h), int(bottom * h), int(left * w), int(right * w)
        flow = cv2.calcOpticalFlowFarneback(a[t:bt, l:r], b[t:bt, l:r], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        moving = np.hypot(flow[..., 0], flow[..., 1]) > MOVING
        dys.append(flow[..., 1][moving])
    moved = np.concatenate(dys)
    return float(moved.mean()) if moved.size else 0.0


def check(frames: list) -> dict:
    """The verdict for one step's frames (BGR arrays at FPS, starting at the command)."""
    frames = frames[int(SKIP_S * FPS)::STRIDE]
    dys = [d for d in (road_flow(p, c) for p, c in zip(frames, frames[1:])) if d is not None]
    if len(dys) < FPS // STRIDE:
        return {"verdict": Motion.TOO_FEW.value, "pairs": len(dys)}
    share = sum(d > MIN_DY for d in dys) / len(dys)
    median = sorted(dys)[len(dys) // 2]
    verdict = Motion.FORWARD if median > MIN_DY else Motion.REVERSE if median < -MIN_DY else Motion.UNCLEAR
    return {"verdict": verdict.value, "pairs": len(dys), "forward_share": round(share, 2), "median_dy": round(median, 2)}


def main(argv: list[str]) -> int:
    import cv2
    for d in argv:
        frames = [cv2.imread(str(p)) for p in sorted(Path(d).glob("*.jpg"))]
        print(json.dumps({"dir": d, **check([f for f in frames if f is not None])}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
