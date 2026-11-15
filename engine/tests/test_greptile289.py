"""The driving-car guard of the final reader (engine/final_reader.py drives_without_template), Greptile on #289.

Each test is what a user says and what the model replies, through the real FinalReadingService, and what the user
then sees: the prompt and the chips, or the edit refused.
"""

import json

from engine import car_template
from engine.final_reader import DriveShot, drive_shot
from engine.tests.test_final_reader import CAR, FakeChat, final, reply, wrapper

STREET = ("A quiet city street with a small cafe on the left. A red car is parked at the curb on the right side of the "
          "street, still. Bright midday sunshine. Cinematic, highly detailed, realistic, sharp focus.")
SIDE = ("A red sports car seen exactly side-on, the whole length of the car from the side, front wheels and rear wheels "
        "both in view, its nose pointing to the left edge of the frame, drives fast from right to left along a straight "
        "two-lane country road, moving forward. The camera moves alongside the car in a smooth side tracking shot at "
        "the same speed, level with the car. Green fields and rows of trees line the far side of the road. Bright "
        "midday sunshine, crisp shadows. Cinematic, highly detailed, realistic, sharp focus, motion blur on the road.")
CITY = car_template.world_prompt("a car", car_template.WORLDS["neon city"])


def chips(out):
    return [c["label"] for c in out["state"]["changes"]]


def test_a_camera_zoom_on_a_parked_car_is_used():
    # Greptile #289 P1: "The camera zooms in" next to a parked car was read as a driving car and refused.
    zoom = STREET.replace("Bright midday", "The camera zooms in slowly on the cafe. Bright midday")
    svc = wrapper(FakeChat({"kill the tree's leaves": reply(STREET, ["city street", "parked car"]),
                            "zoom in on the cafe": reply(zoom, ["city street", "parked car", "zoom in"])}))
    svc.handle(final("kill the tree's leaves"))
    out = svc.handle(final("zoom in on the cafe"))
    assert out["steer"]["prompt"] == zoom and svc.committed_prompt == zoom
    assert chips(out) == ["city street", "parked car", "zoom in"]
    clause = STREET.replace("still.", "still, the camera zooming in on it.")
    assert drive_shot(clause) is DriveShot.NONE  # the camera's verb in the car's own sentence


def test_a_camera_sentence_with_a_car_driving_in_it_is_still_a_driving_car():
    # Greptile #291 P1: only the camera's own verb is ignored, not the car's action after it.
    backward = "A quiet city street at night. The camera tracks a car driving backward down the street."
    say = "i now want a city at night with a car driving on the road"
    chat = FakeChat({say: lambda user, n: reply(CITY if "rejected" in user else backward, ["city at night"],
                                                new_scene=True)})
    out = wrapper(chat).handle(final(say))
    assert drive_shot(backward) is DriveShot.OTHER
    assert out["steer"]["prompt"] == CITY and len(chat.calls) == 2  # refused, asked again, the measured shot used


def test_a_side_view_stays_usable_for_later_edits():
    # Greptile #289 P1: after "show me a side view" was used, "make it rain" on that side view was refused.
    rain = SIDE.replace("Bright midday sunshine, crisp shadows.", "Heavy rain falling, dark grey storm clouds.")
    chat = FakeChat({"show me a side view of the car": reply(SIDE, ["side view"]),
                     "make it rain": reply(rain, ["side view", "rain"])})
    svc = wrapper(chat)
    svc.handle(final(CAR))  # the measured template: no call
    assert car_template.is_template(svc.committed_prompt) and not chat.calls
    assert svc.handle(final("show me a side view of the car"))["steer"]["prompt"] == SIDE
    out = svc.handle(final("make it rain"))
    assert out["steer"]["prompt"] == rain and chips(out) == ["side view", "rain"]
    assert len(chat.calls) == 2  # used at once, no retry


def test_a_new_driving_scene_must_use_the_whole_measured_shot():
    # Greptile #289 P2: "moving forward away from the camera" with a static camera passed for a new scene.
    static = "A car drives along a city street at night, moving forward away from the camera. The camera stays static."
    say = "i now want a city at night with a car driving on the road"

    def answer(user, n_messages):
        return reply(CITY if "rejected" in user else static, ["city at night", "car"], new_scene=True)

    chat = FakeChat({say: answer})
    out = wrapper(chat).handle(final(say))
    assert out["steer"]["prompt"] == CITY and chips(out) == ["city at night", "car"]
    assert len(chat.calls) == 2 and "car template" in json.dumps(chat.calls[1][1]["rejected"])
    assert drive_shot(static) is DriveShot.FORWARD and drive_shot(CITY) is DriveShot.TEMPLATE
    # Live reply (acceptance/sonnet_replies.json): another vehicle word with the camera sentence's "the car" is the shot.
    sedan = CITY.replace("A car,", "A dark red sedan,")
    assert sedan != CITY and drive_shot(sedan) is DriveShot.TEMPLATE
    # "make it slow" rewrites the speed words, not the shot.
    slow = CITY.replace("drives fast", "drives slowly").replace("same speed", "same slow speed")
    assert slow != CITY and drive_shot(slow) is DriveShot.TEMPLATE


def test_a_view_change_on_the_template_keeps_its_forward_motion_and_carries_on():
    # Unchanged by the fix: "zoom out" on the car template may rewrite the camera, then later edits keep it.
    wide = CITY.replace(car_template.camera_sentence("a car"),
                        "A wide establishing shot from high above and far behind, the car small in the frame.")
    rain = wide.replace("light rain falling", "heavy rain pouring down")
    say = "i now want a city at night with a car driving on the road"
    svc = wrapper(FakeChat({say: reply(CITY, ["city at night"], new_scene=True),
                            "zoom out": reply(wide, ["city at night", "wide shot"]),
                            "more rain": reply(rain, ["city at night", "wide shot"])}))
    assert svc.handle(final(say))["steer"]["prompt"] == CITY
    assert svc.handle(final("zoom out"))["steer"]["prompt"] == wide
    assert svc.handle(final("more rain"))["steer"]["prompt"] == rain
