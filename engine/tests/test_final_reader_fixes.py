"""The final reader fixes of issue #117, on CPU with a fake chat that replays the reply shapes real Sonnet gave.

Every test plays sentences through the real commands.rewriter (validate_rewrite) and the real CommandService,
as the demo prompts regeneration does.
"""

import json
import logging

from commands.composer import compose
from commands.service import CommandService
from engine.final_reader import (FORWARD_MOTION, LANE_CUE, STANDIN, VEHICLE_STYLE, FinalReader, FinalReaderState,
                                 prompt_spec, standin_validate)

WINTER = {"op": "add", "label": "winter", "kind": "season",
          "phrase": "deep winter, snow covering the ground, frost on bare branches", "strength": 1.0}
NIGHT = {"op": "add", "label": "night", "kind": "time", "phrase": "deep night, dark sky, soft cold moonlight",
         "strength": 1.0}
CAR = {"op": "add", "label": "red car", "kind": "extra",
       "phrase": "a red car parked beside the oak tree in the centre of the frame, facing the camera, large and clearly visible",
       "strength": 1.0}
LION = {"op": "add", "label": "lion", "kind": "extra",
        "phrase": "a large male lion standing in the foreground at the left of the frame, facing the camera", "strength": 1.0}
DESERT = {"op": "add", "label": "desert", "kind": "world", "phrase": "a desert", "strength": 1.0}


def reply(ops=(), details=""):
    return json.dumps({"blocked": False, "operations": list(ops), "details": details})


class ReplayChat:
    """Answers each sentence with its next reply in `replies[sentence]` (the last one repeats), and records the
    user messages, so a test sees what a retry sends back."""

    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = {k: list(v) if isinstance(v, list) else [v] for k, v in replies.items()}
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append((system, user, messages))
        queue = self.replies[user["sentence"]]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def play(chat, sentences, base=None):
    """The sentences through the reader and the real CommandService; the list of results."""
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec())
    assert reader.spec.name == "commands.rewriter"
    svc = CommandService(base, semantic_reader=lambda *a: None)
    out = []
    for s in sentences:
        res = reader(s, svc.snapshot())
        assert res is not None and not res.get("blocked"), (s, res)
        svc.apply_final(s, res["operations"], None)
        out.append({**res, "scene": svc.snapshot(), "canonical": compose(svc.snapshot())})
    return reader, out


def test_rejected_details_are_logged_with_the_reason_and_repaired(caplog):
    # (a) Real shape: Sonnet's details for "make it night" had "cold still air" ("still" is a time-order word for
    # validate_rewrite) and "a few distant stars" (an object, and "distant" moves things). Before, the whole details
    # were dropped without a word and the prompt was the bare composed text.
    details = "pale silver light on the snow, cold still air, a few distant stars, frost glittering on the ground"
    chat = ReplayChat({"make it winter": reply([WINTER], "powdery snow drifts, soft diffuse winter light"),
                       "make it night": reply([NIGHT], details)})
    with caplog.at_level(logging.WARNING, logger="engine.final_reader"):
        reader, (_, night) = play(chat, ["make it winter", "make it night"])
    assert night["prompt"].endswith("; pale silver light on the snow, frost glittering on the ground")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("'still'" in w and "cold still air" in w for w in warnings), warnings
    assert any("'stars'" in w and "distant stars" in w for w in warnings), warnings
    assert reader.stats()["details_repaired"] == 1 and reader.stats()["details_rejected"] == 0


def test_too_long_details_are_cut_at_a_clause_and_the_budget_is_sent(monkeypatch):
    # (a) Real shape: a long composed prompt (car, lion, winter, night) plus 46 words of details went over the
    # validator's total word limit and was dropped whole. Now the last clauses are cut until it fits.
    # The limit is pinned at its old 100 words so this short scene still goes over it.
    from commands import rewriter
    monkeypatch.setattr(rewriter, "MAX_TOTAL_WORDS", 100)
    long = ", ".join(["frost-dusted ground", "crisp cold air", "soft blue-silver glow", "fine icy sparkle on every "
                      "surface", "deep cool shadows", "muted cool colours", "glittering crust of old snow",
                      "pale grey light on the frozen grass", "cold blue tones on every surface"])
    chat = ReplayChat({"make it winter": reply([WINTER], "powdery snow drifts"), "add a red car": reply([CAR], long),
                       "make it night": reply([NIGHT], long), "add a lion": reply([LION], long)})
    _, results = play(chat, ["make it winter", "add a red car", "make it night", "add a lion"])
    lion = results[-1]["prompt"]
    assert "frost-dusted ground, crisp cold air" in lion and len(lion.split()) <= 100
    assert all(isinstance(c[1]["details_max_words"], int) and 12 <= c[1]["details_max_words"] <= 45 for c in chat.calls)


def test_fully_rejected_details_are_asked_again_with_the_reason():
    # (a) Every clause is refused ("haze" needs fog, "wildflowers" is an object): one retry, told why.
    bad = reply([WINTER], "faint haze over the fields, wildflowers on the verges")
    good = reply([WINTER], "powdery snow drifts across open ground, soft diffuse winter light")
    chat = ReplayChat({"make it winter": [bad, good]})
    reader, (winter,) = play(chat, ["make it winter"])
    assert winter["prompt"].endswith("; powdery snow drifts across open ground, soft diffuse winter light")
    assert len(chat.calls) == 2 and reader.stats()["retries"] == 1
    retry = chat.calls[1][1]
    assert any("'haze'" in r for r in retry["rejected"]) and any("'wildflowers'" in r for r in retry["rejected"])
    assert chat.calls[1][2][1] == {"role": "assistant", "content": bad}  # the earlier reply is shown back


def test_a_desert_drops_winter_and_object_phrases_carry_no_world_words():
    # (b) Real shape: after "make it winter", "add a red car" gave "a red car parked on the snowy ground" and the
    # desert kept "deep winter, snow covering the ground". A world change now clears the earlier season and
    # weather (commands.state, #115); an object phrase with world words is asked again.
    snowy = dict(CAR, phrase="a red car parked on the snowy ground in the centre of the frame, large and clearly visible")
    chat = ReplayChat({"make it winter": reply([WINTER], "powdery snow drifts"),
                       "add a red car": [reply([snowy], "pale snow-dusted ground"), reply([CAR], "pale snow-dusted ground")],
                       "make it night": reply([NIGHT], "pale silver light on the sand"),
                       "make it a desert": reply([DESERT], "warm rippled sand, fine golden grain, dry cracked earth")})
    _, results = play(chat, ["make it winter", "add a red car", "make it night", "make it a desert"])
    retry = chat.calls[2][1]
    assert retry["sentence"] == "add a red car" and any("world words" in r and "'snowy'" in r for r in retry["rejected"])
    desert = results[-1]
    assert [c["kind"] for c in desert["scene"]["changes"]] == ["extra", "time", "world"]  # winter is gone, night stays
    assert "snow" not in desert["prompt"] and "winter" not in desert["prompt"]
    assert CAR["phrase"] in desert["prompt"] and "sand dunes" in desert["prompt"]


def test_a_season_said_with_the_place_stays():
    # commands.state clears season and weather on a world add (#115); the engine applies the world first, so a
    # reply that lists winter before the desert still keeps the snow the sentence asked for.
    chat = ReplayChat({"make it winter": reply([WINTER], "powdery snow drifts"),
                       "take me to a snowy desert": reply([WINTER, DESERT], "pale sand under a thin crust of snow")})
    _, results = play(chat, ["make it winter", "take me to a snowy desert"])
    assert {c["kind"] for c in results[-1]["scene"]["changes"]} == {"world", "season"}


def test_a_long_positioned_vehicle_phrase_is_valid_and_gets_the_lane_cue():
    # (c) Real shape: "add a red car racing on a road" returned None: the phrase had 151 characters and the
    # operation check allowed 120. The final reader allows 200, and a vehicle on a road gets LANE_CUE.
    racing = {"op": "add", "label": "racing car", "kind": "extra", "strength": 1.0,
              "phrase": "a sleek red race car speeding along the road in the middle ground, moving from right to left "
                        "across the frame, kicking up dust behind it"}
    assert 120 < len(racing["phrase"]) <= 200
    chat = ReplayChat({"add a red car racing on a road": reply([racing], "warm dry air, fine dust")})
    _, (res,) = play(chat, ["add a red car racing on a road"])
    assert res["operations"][0]["phrase"] == racing["phrase"] + ", " + LANE_CUE


def test_a_too_long_phrase_is_asked_again_with_its_length():
    huge = {"op": "add", "label": "lion", "kind": "extra", "strength": 1.0, "phrase": "a large lion " + "very " * 60}
    chat = ReplayChat({"add a lion": [reply([huge], "soft grass"), reply([LION], "soft green grass, warm light")]})
    _, (res,) = play(chat, ["add a lion"])
    assert res["operations"] == [LION]
    assert any("characters, the limit is 240" in r for r in chat.calls[1][1]["rejected"])


def test_vehicle_wording_has_the_lane_markings_cue():
    # (d)
    assert LANE_CUE == "white dashed lane markings rush toward the viewer" and FORWARD_MOTION.endswith(LANE_CUE)
    chat = ReplayChat({"x": reply()})
    reader = FinalReader(FinalReaderState.ON, chat=chat, spec=prompt_spec())
    assert LANE_CUE in reader.system_prompt


def test_a_new_object_without_size_or_position_is_asked_again_and_then_kept_after_the_subject():
    # (e) Measured on GPU: "a lion standing in the snow" is often missing or tiny. The phrase needs size and
    # position words; it stays word for word right after the subject in every later prompt.
    vague = dict(LION, phrase="a lion standing proudly")
    chat = ReplayChat({"add a lion": [reply([vague], "soft green grass"), reply([LION], "soft green grass, warm light")],
                       "make it winter": reply([WINTER], "powdery snow drifts, soft diffuse light"),
                       "make it night": reply([NIGHT], "pale silver light on the snow")})
    _, results = play(chat, ["add a lion", "make it winter", "make it night"])
    reasons = chat.calls[1][1]["rejected"]
    assert any("no size word" in r for r in reasons) and any("no position words" in r for r in reasons)
    subject = "a wide shot of a meadow with a single oak tree; "
    assert results[0]["prompt"].startswith(subject + LION["phrase"] + "; ")
    for res in results[1:]:  # winter comes before the objects; the lion stays verbatim right after it
        assert res["prompt"].startswith(subject + WINTER["phrase"] + "; " + LION["phrase"] + "; "), res["prompt"]


def test_position_phrases_stay_verbatim_and_details_never_place_things():
    # (g) Measured on GPU: "make it night" re-framed the shot. Details with position words are cut; the car's
    # "parked beside the oak tree" phrase is in every later prompt, word for word, in the same place.
    chat = ReplayChat({"add a red car": reply([CAR], "soft green grass, warm light"),
                       "make it winter": reply([WINTER], "powdery snow drifts, soft diffuse light"),
                       "make it night": reply([NIGHT], "the oak tree far in the distance, pale silver light on the snow")})
    _, results = play(chat, ["add a red car", "make it winter", "make it night"])
    subject = "a wide shot of a meadow with a single oak tree; "
    assert results[0]["prompt"].startswith(subject + CAR["phrase"] + "; ")
    assert all(r["prompt"].startswith(subject + WINTER["phrase"] + "; " + CAR["phrase"] + "; ") for r in results[1:])
    assert "distance" not in results[-1]["prompt"] and results[-1]["prompt"].endswith("pale silver light on the snow")


def test_no_road_words_in_a_scene_without_a_road():
    # (h) Measured on GPU: the meadow grew a road or tyre tracks with "motion blur on the road" and road details.
    chat = ReplayChat({"add a red car": reply([CAR], "soft green grass, faint tyre tracks, warm light"),
                       "make it night": reply([NIGHT], "a quiet road in moonlight, pale silver light on the grass")})
    _, results = play(chat, ["add a red car", "make it night"])
    for r in results:
        low = r["prompt"].lower()
        assert not any(w in low for w in ("road", "tyre", "motion blur", "lane")), r["prompt"]
    assert results[-1]["prompt"].endswith("pale silver light on the grass")
    # the stand-in says "motion blur on the road" only for a vehicle on a road
    assert VEHICLE_STYLE not in standin_validate("a red car parked in a meadow", {"details": "green grass."})
    assert VEHICLE_STYLE in standin_validate("a red car driving along a road", {"details": "green fields."})
    assert STANDIN.name == "engine stand-in"


def test_a_second_red_car_gets_its_own_label_and_the_first_one_stays():
    # Real shape: "add a red car racing on a road" with a red car in the scene came back with the label "red car",
    # and commands.state replaced the first car with it. A "second" object now gets its own label.
    second = {"op": "add", "label": "red car", "kind": "extra", "strength": 1.0,
              "phrase": "a second small red car racing in the next lane at the right of the frame"}
    chat = ReplayChat({"add a red car": reply([CAR], "soft green grass, warm light"),
                       "add a red car racing on a road": reply([second], "warm dry air, fine dust")})
    _, (_, res) = play(chat, ["add a red car", "add a red car racing on a road"])
    assert [c["label"] for c in res["scene"]["changes"]] == ["red car", "second red car"]
    assert CAR["phrase"] in res["prompt"] and second["phrase"] + ", " + LANE_CUE in res["prompt"]


def test_the_season_comes_before_a_new_object_and_a_desert_does_not_bring_it_back():
    # Measured on GPU: "add a red car" after "make it winter" lost the winter (green grass) when the car phrase
    # came first. The season and weather phrases now come right after the subject, before the objects.
    chat = ReplayChat({"make it winter": reply([WINTER], "powdery snow drifts"),
                       "add a red car": reply([CAR], "pale frost-dusted grass, crisp cold air"),
                       "make it a desert": reply([DESERT], "warm rippled sand, dry cracked earth")})
    _, (_, car, desert) = play(chat, ["make it winter", "add a red car", "make it a desert"])
    assert car["prompt"].startswith(
        "a wide shot of a meadow with a single oak tree; " + WINTER["phrase"] + "; " + CAR["phrase"] + "; ")
    assert desert["prompt"].startswith("a wide shot of a single oak tree; " + CAR["phrase"] + "; ")
    assert WINTER["phrase"] not in desert["prompt"]


def test_the_main_object_is_not_small_and_a_later_one_may_be():
    # Measured on GPU: "a small red car ... at the right of the frame" was small and far right in every frame.
    small = dict(CAR, phrase="a small red car parked beside the oak tree at the right of the frame, facing the camera")
    chat = ReplayChat({"add a red car": [reply([small], "soft green grass"), reply([CAR], "soft green grass")],
                       "add a bird": reply([{"op": "add", "label": "bird", "kind": "extra", "strength": 1.0,
                                             "phrase": "a small bird perched on a branch at the left of the frame"}],
                                           "soft green grass")})
    _, (car, bird) = play(chat, ["add a red car", "add a bird"])
    assert any("the main object" in r and "'small'" in r for r in chat.calls[1][1]["rejected"])
    assert car["operations"] == [CAR] and len(chat.calls) == 3  # the bird is not the main object: no retry
    system = " ".join(chat.calls[0][0].split())
    assert "a red car parked beside the oak tree in the centre of the frame, facing the camera, large and clearly visible" in system
    assert "a large male lion standing in the foreground at the left of the frame, facing the camera" in system
