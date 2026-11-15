"""Named people (engine/people.py) through the final reader and the real CommandService, with real sentences."""

import json

from commands.composer import compose
from commands.service import CommandService
from commands.state import SceneState
from engine import car_template as ct
from engine.final_reader import FinalReader, FinalReaderState
from engine.people import PEOPLE, for_model
from engine.reader import FinalReadingService

JENSEN = PEOPLE[0].description
LEFT = f"{JENSEN}, a large figure standing in the foreground at the left of the frame, facing the camera"
LION = {"op": "add", "label": "lion", "kind": "extra", "strength": 1.0,
        "phrase": "a large male lion standing in the foreground at the left of the frame, facing the camera"}
CAR = {"op": "add", "label": "red car", "kind": "extra", "strength": 1.0,
       "phrase": "a large red car parked beside the oak tree in the centre of the frame, facing the camera"}


class Chat:
    """Replies by the sentence the model reads; records what it was sent. A missing sentence fails the call."""
    model = "fake-sonnet"

    def __init__(self, replies):
        self.replies = replies
        self.sent = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.sent.append(user)
        return json.dumps(self.replies[user["sentence"]])


def reply(prompt, chips=()):
    return {"blocked": False, "understood": True, "new_scene": False, "prompt": prompt, "chips": list(chips)}


WITH_HIM = f"A wide shot of a meadow with a single oak tree, {JENSEN} standing beside it, large and clearly visible."


def final(text):
    return {"type": "final", "text": text, "t_client": 1}


def service(replies):
    chat = Chat(replies)
    reader = FinalReader(FinalReaderState.ON, chat=chat)
    return chat, reader, FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)


def labels(out):
    return [c["label"] for c in out["state"]["changes"]]


BARE = "A wide shot of a meadow with a single bare oak tree."


def free_service(replies):
    """A service whose scene Sonnet already wrote ("kill the tree's leaves"): edits of it go to the model."""
    chat, reader, svc = service({"kill the tree's leaves": reply(BARE, ["bare branches"]), **replies})
    svc.handle(final("kill the tree's leaves"))
    return chat, reader, svc


def test_add_jensen_huang_on_a_free_scene_sends_the_description_and_keeps_his_name_on_the_chip():
    chat, _, svc = free_service({for_model("add jensen huang"): reply(WITH_HIM, ["man in leather jacket"])})
    out = svc.handle(final("add jensen huang"))
    assert chat.sent[1]["sentence"] == f"add {JENSEN}" and chat.sent[1]["person_chip"] == "jensen huang"
    assert "jensen huang" in labels(out)  # the engine adds his chip when the model names him another way
    assert out["steer"]["prompt"] == WITH_HIM


def test_the_nvidia_ceo_is_the_same_person():
    chat, _, svc = free_service({for_model("add the nvidia ceo"): reply(WITH_HIM, ["jensen huang"])})
    out = svc.handle(final("Add the NVIDIA CEO"))
    assert chat.sent[1]["sentence"] == f"add {JENSEN}" and "jensen huang" in labels(out)


def test_a_bare_name_is_an_add():
    # Live on #273, 08:33 UTC: "jensen huang" alone came back not understood.
    assert for_model("Jensen Huang.") == f"add {JENSEN}" and for_model("i want jensen") == f"add {JENSEN}"
    assert for_model("jensen dancing") == "jensen dancing"  # more than the name: the words stay


def test_a_sentence_that_only_adds_or_removes_him_is_the_engines_on_a_composed_scene_with_no_call():
    for text in ("jensen huang", "jensen", "add jensen", "add jensen huang", "Add the NVIDIA CEO", "i want jensen"):
        chat, reader, svc = service({})
        out = svc.handle(final(text))
        assert labels(out) == ["jensen huang"] and LEFT in out["steer"]["prompt"], text
        assert chat.sent == [] and reader.stats()["person_hits"] == 1, text
        out = svc.handle(final("remove jensen"))
        assert labels(out) == [] and chat.sent == [] and reader.stats()["person_hits"] == 2, text


def test_a_sentence_with_more_than_him_is_not_the_engines():
    from engine.people import whole_sentence
    meadow = SceneState().snapshot()
    for text in ("add jensen huang and a dog", "jensen dancing on the moon", "make jensen wave"):
        assert whole_sentence(text, meadow) is None, text
    assert whole_sentence("add jensen huang next to the car", meadow)["phrase"].endswith("next to the car, facing the camera")


def test_he_takes_a_free_spot_when_the_lion_holds_the_left():
    from engine.people import person_add
    s = SceneState()
    s.apply(LION)
    assert person_add("add jensen huang", s.snapshot())["phrase"] == \
        f"{JENSEN}, a large figure standing in the foreground at the right of the frame, facing the camera"


def test_put_jensen_huang_next_to_the_car():
    from engine.people import person_add
    s = SceneState()
    s.apply(CAR)
    assert person_add("put jensen huang next to the car", s.snapshot())["phrase"] == \
        f"{JENSEN}, a large figure standing next to the car, facing the camera"


def test_remove_jensen_huang_and_undo_find_him():
    chat, _, svc = free_service({for_model("add jensen huang"): reply(WITH_HIM, ["jensen huang"]),
                                 "remove jensen huang": reply(BARE, ["bare branches"])})
    svc.handle(final("add jensen huang"))
    out = svc.handle(final("remove jensen huang"))
    assert chat.sent[-1]["sentence"] == "remove jensen huang"  # the words stay, so the chip label matches
    assert "jensen huang" not in labels(out) and JENSEN not in out["steer"]["prompt"]
    back = svc.handle({"type": "tap", "action": "undo"})
    assert "jensen huang" in labels(back) and back["steer"]["prompt"] == WITH_HIM


def test_remove_jensen_huang_through_the_dictionary_when_the_model_is_down():
    _, _, svc = service({})  # every call fails: the engine's add, then the dictionary's remove
    svc.handle(final("add jensen huang"))
    out = svc.handle(final("remove jensen huang"))
    assert labels(out) == []


def test_add_jensen_huang_with_no_model_answer_is_the_engines_add():
    _, reader, svc = service({})  # every call fails
    out = svc.handle(final("add jensen huang and a dog"))  # not only him: the model is asked, and fails
    assert "jensen huang" in labels(out) and LEFT in out["steer"]["prompt"]
    assert reader.stats()["errors"] == 1


def test_add_jensen_huang_on_the_car_scene_is_in_the_template():
    country = ct.WORLDS["country"]
    _, _, svc = service({})  # the model is down: the engine's add, in the template
    svc.handle(final("new scene of a red sports car driving along a country road"))
    out = svc.handle(final("add jensen huang"))
    assert labels(out) == ["jensen huang"]
    assert out["steer"]["prompt"] == ct.world_prompt("a red sports car", country, [LEFT])


def test_the_pinned_demo_plays_every_entry_and_jensen_huang_with_no_call():
    from engine.final_reader import DEFAULT_DEMO_PROMPTS, load_pinned
    entries = json.loads(open(DEFAULT_DEMO_PROMPTS).read())
    pinned = load_pinned(DEFAULT_DEMO_PROMPTS)
    assert len(pinned) == len(entries)
    reader = FinalReader(FinalReaderState.ON, chat=Chat({}), pinned=pinned)
    svc = FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader)
    base = compose(SceneState().snapshot())
    seen = {}  # composed prompt -> the words the model sees for it: the first rich prompt of that scene
    for e in entries:
        if e["prompt_before"] == base and compose(svc._svc.snapshot()) != base:
            svc.handle(final("reset"))  # the file starts a chain again from the meadow
        assert compose(svc._svc.snapshot()) == e["prompt_before"], e["sentence"]
        out = svc.handle(final(e["sentence"]))
        assert out["steer"]["prompt"] == seen.setdefault(compose(svc._svc.snapshot()), e["prompt"]), e["sentence"]
        if e["sentence"] == "add jensen huang":
            assert labels(out)[-1] == "jensen huang" and JENSEN in out["steer"]["prompt"].lower()
    assert reader.stats()["calls"] == 0 and reader.stats()["pinned_hits"] == len(entries)
    assert [e["sentence"] for e in entries].count("add jensen huang") >= 2


def test_person_remove_only_for_a_sentence_that_is_just_that_removal():
    from engine.people import person_remove
    snap = {"changes": [{"id": "c1", "label": "jensen huang"}, {"id": "c2", "label": "dog"}]}
    assert person_remove("remove jensen", snap) == {"op": "remove", "id": "c1"}
    assert person_remove("please remove the nvidia ceo", snap) == {"op": "remove", "id": "c1"}
    assert person_remove("remove the dog near jensen", snap) is None  # the dog is the target
    assert person_remove("remove jensen huang and add a dog", snap) is None  # the keyword table reads both clauses
