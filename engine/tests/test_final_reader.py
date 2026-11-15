"""The Sonnet final reader (engine/final_reader.py) and its CommandService wrapper (engine/reader.py), issue #271.

A fake chat client stands in for Foundry: no network. The wrapper runs against the real commands.service.CommandService,
so a change to its private fields fails here, not in the live demo.
"""

import json
import threading
import time

import httpx
import pytest

from commands.service import CommandService
from commands.state import SceneState
from engine import car_template
from engine.final_reader import (CacheFile, FinalReader, FinalReaderState, Outcome, Verdict, call_options, check_reply,
                                 MAX_PROMPT_WORDS, fingerprint, fit_prompt, load, reply_chips)
from engine.foundry import FoundryMessages
from engine.reader import FinalReadingService, load_reader

OAK = "a wide shot of a meadow with a single oak tree"
CAR = "new scene of a red sports car driving along a country road"
WINTER = {"op": "add", "label": "winter", "kind": "season", "phrase": "snow on the ground", "strength": 1.0}
LION = {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion in the grass", "strength": 1.0}


def reply(prompt="", chips=(), understood=True, blocked=False, new_scene=False):
    return json.dumps({"blocked": blocked, "understood": understood, "new_scene": new_scene, "prompt": prompt,
                       "chips": list(chips)})


class FakeChat:
    """Answers `replies[sentence]` (a reply string, an exception, or a function of the user message). A sentence in
    `slow` waits for its event first. Any other sentence: not understood."""

    model = "fake-sonnet"

    def __init__(self, replies=None, slow=None):
        self.replies = replies or {}
        self.slow = slow or {}
        self.calls = []

    def complete(self, system, messages):
        user = json.loads(messages[-1]["content"])
        self.calls.append((system, user, messages))
        if user["sentence"] in self.slow:
            self.slow[user["sentence"]].wait(10)
        answer = self.replies.get(user["sentence"], reply(understood=False))
        if callable(answer):
            answer = answer(user, len(messages))
        if isinstance(answer, Exception):
            raise answer
        return answer


def reader(chat, **kwargs):
    return FinalReader(FinalReaderState.ON, chat=chat, **kwargs)


def snap(*ops, base=OAK):
    s = SceneState(base)
    for op in ops:
        s.apply(op)
    return s.snapshot()


def final(text, t=1):
    return {"type": "final", "text": text, "t_client": t}


def wrapper(chat, **kwargs):
    return FinalReadingService(CommandService(semantic_reader=lambda *a: None), reader(chat, **kwargs))


def car_prompt(world="country", objects=()):
    return car_template.world_prompt("a red sports car", car_template.WORLDS[world], objects)


# --- the light check --------------------------------------------------------------------------------------

def test_a_reply_becomes_one_free_scene_operation():
    a = check_reply(json.loads(reply("A bare oak tree on cracked dry earth.", ["Bare Tree", "arid ground"])),
                    {"car_scene": False})
    assert a.verdict is Verdict.ACCEPTED
    assert a.entry.operations == ({"op": "free_scene", "prompt": "A bare oak tree on cracked dry earth.",
                                   "chips": ["bare tree", "arid ground"]},)


@pytest.mark.parametrize("answer, verdict", [
    (reply(blocked=True), Verdict.BLOCKED),
    (reply(understood=False), Verdict.NOT_UNDERSTOOD),
    (reply("   "), Verdict.NOT_UNDERSTOOD),
    ("[1, 2]", Verdict.INVALID),
])
def test_blocked_not_understood_and_unusable_replies(answer, verdict):
    assert check_reply(json.loads(answer), {}).verdict is verdict


def test_a_long_prompt_is_cut_at_sentences_not_refused():
    long = " ".join(f"Sentence {n} has five words." for n in range(80))  # 400 words
    prompt, cut = fit_prompt(long)
    assert len(prompt.split()) <= MAX_PROMPT_WORDS and prompt.startswith("Sentence 0 has") and prompt.endswith(".") and cut
    style_kept, _ = fit_prompt(car_prompt() + " " + long.replace("Sentence", "Extra") + " " + car_template.STYLE)
    assert style_kept.endswith(car_template.STYLE) and car_template.is_template(style_kept)
    a = check_reply(json.loads(reply(long, ["x"])), {})
    assert a.verdict is Verdict.REPAIRED and len(a.entry.prompt.split()) <= MAX_PROMPT_WORDS


def test_chips_are_cleaned_never_a_reason_to_refuse():
    assert reply_chips(["A Very Long Chip Label With Many Words", "lion", "lion", 3, ""], "jensen huang") == \
        ["a very long chip label", "lion", "jensen huang"]
    assert reply_chips("lion") == []


def test_a_car_scene_keeps_the_template_unless_the_view_or_the_scene_changes():
    car = {"car_scene": True}
    beach = car_prompt("ocean coast")
    assert check_reply(json.loads(reply(beach, ["beach"])), car).verdict is Verdict.ACCEPTED
    on_sand = "A red sports car parked on the sand of a wide beach, facing the waves."
    broken = check_reply(json.loads(reply(on_sand, ["beach"])), car)
    assert broken.verdict is Verdict.REPAIRED and "car scene" in broken.reasons[0]  # used, logged, asked again
    road = car_prompt().replace("a straight two-lane country road", "the soft sand of a beach")
    assert "paved road" in check_reply(json.loads(reply(road)), car).reasons[0]
    assert check_reply(json.loads(reply(on_sand, new_scene=True)), car).verdict is Verdict.ACCEPTED
    side = ("A red sports car, seen in side profile, drives from left to right along a coastal road. A smooth "
            "tracking shot from the side, the camera moving alongside the car, the ocean behind it.")
    side_view = {**car, "view_change": True, "sentence": "show me a side view of the car"}
    assert check_reply(json.loads(reply(side)), side_view).verdict is Verdict.ACCEPTED
    assert check_reply(json.loads(reply(on_sand)), side_view).verdict is Verdict.REPAIRED
    # Without the words "side view" a driving car keeps the template's forward shot (live: the side view reversed).
    assert check_reply(json.loads(reply(side)), {**car, "view_change": True}).verdict is Verdict.NOT_FORWARD


def test_view_change_words():
    assert car_template.asks_view_change("change the point of view to be facing the ocean. "
                                         "i want to see the car's side profile")
    for text in ("zoom out", "a wider shot", "aerial view please", "low angle", "pan to the left"):
        assert car_template.asks_view_change(text), text
    for text in ("make it winter", "add a pancake", "take me to the beach"):
        assert not car_template.asks_view_change(text), text


def test_the_system_prompt_carries_the_placement_motion_and_camera_rules():
    system = " ".join(reader(FakeChat()).system_prompt.split())
    for rule in ("Return the complete new prompt, never a diff",
                 "for every object say where it is, what it stands or drives on, and which way it faces",
                 "its tail fin and the backs of its engines toward the camera, its nose pointing away toward the horizon",
                 "front wheels and rear wheels both in view",
                 "starts a new scene",
                 "A bare noun phrase",
                 "rewrite the camera wording to match",
                 "the whole length of the car from the side",
                 car_template.WORLDS["winter"].scene):
        assert rule in system, rule


# --- the reader -------------------------------------------------------------------------------------------

def test_the_user_message_is_the_prompt_on_screen_the_chips_and_the_sentence():
    chat = FakeChat({"kill the tree's leaves": reply("A bare oak tree.", ["bare tree"])})
    r = reader(chat)
    out = r("kill the tree's leaves", snap(LION), current_prompt="RICH PROMPT")
    assert out["operations"][0] == {"op": "free_scene", "prompt": "A bare oak tree.", "chips": ["bare tree"]}
    assert chat.calls[0][1] == {"prompt": "RICH PROMPT", "chips": ["lion"], "sentence": "kill the tree's leaves",
                                "car_scene": False}
    free = snap({"op": "free_scene", "prompt": "A bare oak tree.", "chips": ["bare tree"]})
    r("add jensen huang", free)  # on a composed scene the engine adds him with no call (engine/people.py)
    user = chat.calls[1][1]
    assert user["person_chip"] == "jensen huang" and "leather jacket" in user["sentence"]
    r("i want to see the side profile", snap(base="a red sports car"), current_prompt=car_prompt())
    assert chat.calls[2][1]["car_scene"] and chat.calls[2][1]["view_change"]


def test_word_list_blocks_without_a_call_and_the_model_can_block():
    chat = FakeChat({"make them hurt each other": reply(blocked=True)})
    r = reader(chat)
    assert r("show me a naked man", snap()) == {"blocked": True} and not chat.calls
    assert r("make them hurt each other", snap()) == {"blocked": True}
    assert r("make them hurt each other", snap()) == {"blocked": True} and len(chat.calls) == 1  # saved


def test_failed_calls_and_unusable_replies_fall_back_and_are_final_in_the_process():
    chat = FakeChat({"a": "not json", "b": "[]", "f": httpx.ConnectError("down")})
    r = reader(chat)
    for sentence in "abf":
        assert r(sentence, snap()) is None
    s = r.stats()
    assert s["rejects"] == 2 and s["errors"] == 1 and s["retries"] == 2  # unusable replies are asked again once
    assert r("f", snap()) is None and len(chat.calls) == 5  # a fallback is final in this process: no new call


def test_a_reply_that_leaves_the_prompt_on_screen_is_not_used():
    # Regression sweep on #273: a cut took the new object away, the prompt stayed the same and only the chip changed.
    on_screen = "A bare oak tree in a meadow."
    chat = FakeChat({"add a lake": reply(on_screen + "  ", ["bare tree", "lake"])})
    r = reader(chat)
    free = snap({"op": "free_scene", "prompt": on_screen, "chips": ["bare tree"]})
    assert r("add a lake", free, current_prompt=on_screen) is None  # the dictionary path: not understood there
    assert r.stats()["verdicts"] == {"unchanged": 1} and r.stats()["retries"] == 1 and len(chat.calls) == 2


def test_a_fallback_on_a_free_scene_is_asked_again():
    # Review of #273: on a free scene the dictionary refuses edits, so a kept fallback would fail there until restart.
    chat = FakeChat({"f": httpx.ConnectError("down")})
    r = reader(chat)
    free = snap({"op": "free_scene", "prompt": "A bare oak tree.", "chips": ["bare tree"]})
    assert r("f", free) is None and r("f", free) is None and len(chat.calls) == 2


def test_a_repaired_reply_is_asked_again_and_the_better_one_wins():
    good = car_prompt("ocean coast")
    chat = FakeChat({"take me to a sunny beach with palm trees": lambda user, n: reply("A car on the sand.") if n == 1
                     else reply(good, ["beach"])})
    r = reader(chat)
    out = r("take me to a sunny beach with palm trees", snap(base="a red sports car"), current_prompt=car_prompt())
    assert out["prompt"] == good and r.stats()["retries"] == 1
    rejected = json.loads(chat.calls[1][2][-1]["content"])["rejected"]
    assert "car scene" in rejected[0]


def test_a_slow_retry_keeps_the_usable_first_reply():
    gate = threading.Event()

    def answer(user, n):
        if n > 1:
            gate.wait(5)
        return reply("A car on the sand.", ["beach"])
    r = reader(FakeChat({"to the sand": answer}), timeout_s=0.4)
    out = r("to the sand", snap(base="a red sports car"), current_prompt=car_prompt())
    gate.set()
    assert out["prompt"] == "A car on the sand." and r.stats()["timeouts"] == 1


def test_timeout_falls_back_once_and_drops_the_late_answer():
    gate = threading.Event()
    chat = FakeChat({"slow": reply("A slow prompt.")}, slow={"slow": gate})
    r = reader(chat, timeout_s=0.1)
    t0 = time.monotonic()
    assert r("slow", snap()) is None
    assert time.monotonic() - t0 < 0.5 and r.stats()["timeouts"] == 1
    gate.set()
    t0 = time.monotonic()
    while r.stats()["late_dropped"] == 0 and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert r.stats()["late_dropped"] == 1
    assert r("slow", snap()) is None and len(chat.calls) == 1  # no late swap, no new call


def test_cache_file_survives_a_restart(tmp_path):
    path = tmp_path / "sub" / "cache.json"
    chat = FakeChat({"cut the grass": reply("A mown lawn.", ["mown grass"]), "boom": httpx.ConnectError("down"),
                     "make them hurt each other": reply(blocked=True), "hmm": reply(understood=False)})
    r = reader(chat)
    r._file = CacheFile(path, fingerprint(r.system_prompt, chat.model))
    first = r("cut the grass", snap())
    for sentence in ("boom", "make them hurt each other", "hmm"):
        r(sentence, snap())
    data = json.loads(path.read_text())
    assert {e["outcome"] for e in data["entries"]} == {"sonnet", "blocked", "not_understood"}  # no fallback saved
    assert not list(path.parent.glob(".final-reader-*"))
    chat2 = FakeChat({"cut the grass": reply("Something else entirely.")})
    r2 = reader(chat2, cache_file=CacheFile(path, fingerprint(r.system_prompt, chat.model)))
    assert r2("cut the grass", snap()) == first and r2("hmm", snap()) == {"not_understood": True}
    assert not chat2.calls  # never regenerated
    r3 = reader(FakeChat(), cache_file=CacheFile(path, fingerprint(r.system_prompt, "another model")))
    assert r3.stats()["cache_size"] == 0


def test_a_cached_free_scene_is_used_only_on_the_same_scene():
    chat = FakeChat({"cut the grass": reply("A mown lawn.", ["mown grass"])})
    r = reader(chat)
    r("cut the grass", snap())
    lion_weak = snap({**LION, "strength": 0.0})  # the same composed prompt (a zero chip is left out), another scene
    r("cut the grass", lion_weak)
    assert len(chat.calls) == 2


def test_load_states_and_call_options():
    assert load("off").state is FinalReaderState.OFF
    r = load("sonnet", environ={}, cache_path=None)
    assert r.state is FinalReaderState.UNAVAILABLE and "FOUNDRY" in r.stats()["reason"]
    env = {"FOUNDRY_ENDPOINT": "https://madhacks-foundry.openai.azure.com", "FOUNDRY_KEY": "k"}
    r = load("sonnet", environ=env, cache_path=None)
    assert r.state is FinalReaderState.ON and r._chat.model == "claude-sonnet-5-5"
    assert (r._chat.effort, r._chat.thinking, r._chat.max_tokens) == ("low", "between_tools", 1024)
    assert r._chat._url == "https://madhacks-foundry.services.ai.azure.com/anthropic/v1/messages"
    r2 = load("sonnet", environ={**env, "FINAL_READER_EFFORT": "medium", "FINAL_READER_THINKING": "adaptive",
                                 "FINAL_READER_MAX_TOKENS": "4096"}, cache_path=None)
    assert (r2._chat.effort, r2._chat.thinking, r2._chat.max_tokens) == ("medium", "adaptive", 4096)
    with pytest.raises(ValueError):
        call_options({"FINAL_READER_EFFORT": "max"})
    load_reader("standin", r)
    assert r.state is FinalReaderState.UNAVAILABLE and "--reader commands" in r.reason
    assert r("cut the grass", snap()) is None  # UNAVAILABLE behaves as OFF
    with pytest.raises(ValueError):
        load("gpt")


def test_request_body_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": reply("A mown lawn.")}],
                                         "stop_reason": "end_turn"})
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          headers={"x-api-key": "k", "anthropic-version": "2023-06-01"})
    chat = FoundryMessages("https://madhacks-foundry.services.ai.azure.com/anthropic/v1/messages", "k", client=client)
    r = reader(chat)
    assert r("cut the grass", snap())["prompt"] == "A mown lawn."
    body = json.loads(seen[0].content)
    assert set(body) == {"model", "max_tokens", "system", "messages", "thinking", "output_config"}
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 1024
    assert body["thinking"] == {"type": "between_tools"} and body["output_config"] == {"effort": "low"}
    assert body["system"] == r.system_prompt and body["messages"][0]["role"] == "user"


# --- the wrapper: the order of a final, against the real CommandService ------------------------------------

def test_the_order_control_pinned_dictionary_then_sonnet():
    chat = FakeChat({"kill the tree's leaves": reply("A wide shot of a meadow with a bare oak tree.", ["bare tree"])})
    from engine.final_reader import DEFAULT_DEMO_PROMPTS, load_pinned
    svc = wrapper(chat, pinned=load_pinned(DEFAULT_DEMO_PROMPTS))
    out = svc.handle(final("make it winter"))  # pinned: no call
    assert "snow" in out["steer"]["prompt"] and not chat.calls
    out = svc.handle(final("make it night"))  # the dictionary reads every word and writes a time well: no call
    labels = [c["label"] for c in out["state"]["changes"]]
    assert "night" in labels and not chat.calls
    out = svc.handle(final("kill the tree's leaves"))  # the dictionary would add "trees": Sonnet writes it
    assert len(chat.calls) == 1 and chat.calls[0][1]["prompt"] != svc._svc.snapshot()["base"]
    assert "night" in chat.calls[0][1]["prompt"]  # the prompt on screen
    assert out["steer"]["prompt"] == "A wide shot of a meadow with a bare oak tree."
    assert [c["label"] for c in out["state"]["changes"]] == ["bare tree"] and not out["state"]["unmatched"]
    assert svc.committed_prompt == out["steer"]["prompt"] and svc.likely_prompts() == [out["steer"]["prompt"]]
    out = svc.handle(final("undo"))  # control: no call, back to the composed scene and its rich prompt
    assert len(chat.calls) == 1 and [c["label"] for c in out["state"]["changes"]] == labels


def test_object_adds_partial_strengths_and_vehicle_scenes_go_to_sonnet():
    # Regression sweep on #273: the table wrote "add an eagle" as a flock of birds, "add a human" as a group of
    # people, "add a little fog" as "a subtle hint of a thick fog", and on the car scene kept the summer sky in fog.
    from engine.reader import fast_path_ok
    from commands.service import dictionary_reading
    meadow = snap()
    for text in ("add an eagle", "add a human", "add a dog and a cat", "add a little fog"):
        ops = dictionary_reading(text, meadow)
        assert ops is None or not fast_path_ok(ops, meadow, text), text
    for text in ("make it winter", "make it night", "take me to the beach", "make it foggy"):
        assert fast_path_ok(dictionary_reading(text, meadow), meadow, text), text
    car = snap(base="a red sports car driving along a country road")
    assert not fast_path_ok(dictionary_reading("make it foggy", car), car, "make it foggy")


def test_the_fast_path_adds_only_on_the_plain_scene_with_no_amount_word():
    # Regression sweep on 243a1e7 (#285): "make it snow" on the neon city appended the table's winter phrase and kept
    # "light rain falling"; "gradually add some fog" wrote "a thick fog".
    from engine.reader import fast_path_ok
    from commands.service import dictionary_reading
    meadow = snap()
    for text in ("gradually add some fog", "add some rain", "make it light rain", "add more fog", "a lot of snow"):
        ops = dictionary_reading(text, meadow)
        assert ops is None or not fast_path_ok(ops, meadow, text), text
    city = snap({"op": "add", "label": "neon city", "kind": "world", "phrase": "a big city at night, light rain falling",
                 "strength": 1.0})
    for text in ("make it snow", "make it night", "make it foggy", "take me to the beach"):
        ops = dictionary_reading(text, city)
        assert ops is None or not fast_path_ok(ops, city, text), text
    winter = snap({"op": "add", "label": "winter", "kind": "season", "phrase": "winter scenery with snow on the ground",
                   "strength": 1.0})
    for text in ("remove the winter", "undo", "a little more snow"):  # no add: still the table
        ops = dictionary_reading(text, winter)
        assert ops is not None and fast_path_ok(ops, winter, text), text


def test_the_leads_stuck_sequence_changes_the_prompt_every_step():
    # Live 07:27 UTC: after "arid environment" was written into the base, every place change failed.
    steps = ["kill the tree's leaves", "cut the grass", "i want no grass. arid environment", "add a lion",
             "now i want a savannah", "i now want a city in the night with a car driving on the road",
             "city with road and red car driving"]
    city = car_prompt("neon city", ["A large male lion stands on the pavement at the left of the road"])
    answers = {s: reply(f"Prompt {n}: {s}.", [f"chip {n}"]) for n, s in enumerate(steps)}
    answers[steps[5]] = reply(city, ["city at night", "car", "lion"])
    answers[steps[6]] = reply(city.replace("A red sports car", "A bright red sports car"), ["city at night", "red car"])
    svc = wrapper(FakeChat(answers))
    prompts = [svc.committed_prompt]
    for s in steps:
        out = svc.handle(final(s))
        assert out["steer"] is not None and not out["state"]["unmatched"], s
        prompts.append(out["steer"]["prompt"])
    assert len(set(prompts)) == len(prompts)
    assert prompts[-1].startswith("A bright red sports car")


def test_a_timeout_on_a_free_scene_shows_not_understood_never_a_redraw():
    gate = threading.Event()
    chat = FakeChat({"cut the grass": reply("A mown lawn.", ["mown grass"]), "add a dog": reply("x")},
                    slow={"add a dog": gate})
    svc = wrapper(chat, timeout_s=0.2)
    svc.handle(final("cut the grass"))
    out = svc.handle(final("add a dog"))  # read whole by the dictionary, but the scene is free: Sonnet, then late
    gate.set()
    assert out["steer"] is None and out["state"]["unmatched"]
    assert svc.committed_prompt == "A mown lawn."


def test_a_timeout_on_a_composed_scene_falls_back_to_the_dictionary():
    gate = threading.Event()
    chat = FakeChat({"add a red car": reply("x")}, slow={"add a red car": gate})
    svc = wrapper(chat, timeout_s=0.1)
    out = svc.handle(final("add a red car"))
    gate.set()
    assert [c["label"] for c in out["state"]["changes"]] == ["car"]


def test_not_understood_rolls_back_a_guess():
    svc = wrapper(FakeChat())
    guess = svc.handle({"type": "partial", "text": "add a dog", "t_client": 1})
    assert guess["steer"]["provisional"]
    out = svc.handle(final("add a dog with wings, or not, i am unsure"))
    assert out["state"]["unmatched"] and out["steer"]["prompt"] == OAK and not out["steer"]["provisional"]


def test_a_tap_on_a_free_chip_asks_to_remove_it():
    chat = FakeChat({"add a lion": reply("A lion beside a bare oak tree.", ["bare tree", "lion"]),
                     "remove the lion": reply("A bare oak tree.", ["bare tree"])})
    svc = wrapper(chat)
    out = svc.handle(final("add a lion"))
    lion = next(c["id"] for c in out["state"]["changes"] if c["label"] == "lion")
    out = svc.handle({"type": "tap", "action": "remove", "id": lion})
    assert chat.calls[-1][1]["sentence"] == "remove the lion"
    assert out["steer"]["prompt"] == "A bare oak tree." and [c["label"] for c in out["state"]["changes"]] == ["bare tree"]
    out = svc.handle({"type": "tap", "action": "undo"})
    assert out["steer"]["prompt"] == "A lion beside a bare oak tree."


def test_a_new_scene_from_a_free_scene_is_composed_again():
    from engine.final_reader import DEFAULT_DEMO_PROMPTS, load_pinned
    chat = FakeChat({"cut the grass": reply("A mown lawn.", ["mown grass"])})
    svc = wrapper(chat, pinned=load_pinned(DEFAULT_DEMO_PROMPTS))
    svc.handle(final("cut the grass"))
    out = svc.handle(final(CAR))  # measured: no call
    assert len(chat.calls) == 1 and car_template.is_template(out["steer"]["prompt"])
    assert svc._svc.snapshot()["origin"] == "composed"


def test_a_mixed_sentence_the_dictionary_cannot_read_is_still_checked_by_the_model():
    chat = FakeChat({"reset and add a bloody axe": reply(blocked=True)})
    svc = wrapper(chat)
    out = svc.handle(final("reset and add a bloody axe"))
    assert out["steer"] is None and out["state"]["unmatched"] and len(chat.calls) == 1
    out = svc.handle(final("reset and make it night"))  # read whole and written well: the dictionary, no call
    assert [c["label"] for c in out["state"]["changes"]] == ["night"] and len(chat.calls) == 1


def test_engine_reports_the_final_reader():
    from engine.engine import Engine, EngineConfig
    e = Engine(EngineConfig(backend="fake"), lambda b: None, lambda m: None)
    assert e.stats()["final_reader"]["state"] == "off"
    e.stop()


def test_outcomes_are_counted_by_verdict():
    chat = FakeChat({"a": reply("A."), "b": reply(understood=False), "c": "nope"})
    r = reader(chat)
    for s in "abc":
        r(s, snap())
    assert r.stats()["verdicts"] == {"accepted": 1, "invalid": 1, "not_understood": 1}
    assert r.stats()["not_understood"] == 1 and Outcome.NOT_UNDERSTOOD.saved


def test_a_chip_whose_words_left_the_prompt_is_dropped():
    # Regression sweep on 243a1e7 (#285): "make it rain" on a snowy night took every snow word out of the prompt and
    # the "winter" chip stayed with nothing behind it.
    from engine.final_reader import check_reply
    before = ("a wide shot of a meadow with a single oak tree; winter, snow covering the ground and bare branches; "
              "a large male lion; a wide river; night, dark sky lit by a bright full moon")
    after = "A wide shot of a dark meadow at night under heavy rain, a large male lion, a wide river, melting slush."
    user = {"prompt": before, "chips": ["winter", "lion", "night", "river"], "sentence": "make it rain"}
    attempt = check_reply(json.loads(reply(after, ["winter", "lion", "night", "river", "rain"])), user)
    assert attempt.entry.operations[0]["chips"] == ["lion", "night", "river", "rain"]
    # Snow said in other words keeps the chip; a chip never in either prompt, and the named person's chip, stay.
    kept = after.replace("melting slush", "frosty snowdrifts")
    attempt = check_reply(json.loads(reply(kept, ["winter", "rain", "zoom out", "jensen huang"])),
                          {**user, "person_chip": "jensen huang"})
    assert attempt.entry.operations[0]["chips"] == ["winter", "rain", "zoom out", "jensen huang"]
    # Greptile on #286: a longer label keeps the aliases of the thing inside it ("light fog" shown as "a thin mist").
    attempt = check_reply(json.loads(reply("A meadow under a thin mist at night.", ["light fog", "night"])),
                          {"prompt": "a meadow; a light fog drifts low; night", "chips": ["light fog", "night"],
                           "sentence": "make it night"})
    assert attempt.entry.operations[0]["chips"] == ["light fog", "night"]
    # Greptile on #288: the longest keyword wins ("lightning bugs" are fireflies, not the storm's lightning), and a label
    # made only of a filler word ("light") keeps that word as its evidence.
    attempt = check_reply(json.loads(reply("A meadow under a dark thunderstorm with lightning.", ["lightning bugs"])),
                          {"prompt": "a meadow with glowing fireflies", "chips": ["lightning bugs"], "sentence": "storm"})
    assert attempt.entry.operations[0]["chips"] == []
    attempt = check_reply(json.loads(reply("A dark meadow.", ["light"])),
                          {"prompt": "a meadow with a warm light", "chips": ["light"], "sentence": "make it dark"})
    assert attempt.entry.operations[0]["chips"] == []


def test_the_final_reader_waits_3_5_s_by_default():
    # #285: "make it slow" on a long chain timed out 2 of 3 at 3.0 s (median call 1.98 s).
    from engine.engine import EngineConfig
    from engine.final_reader import DEFAULT_TIMEOUT_S
    assert DEFAULT_TIMEOUT_S == 3.5 and EngineConfig.final_timeout_s == 3.5
    assert load("sonnet", chat=FakeChat({}), cache_path=None, demo_prompts=None).timeout_s == 3.5


def test_a_retry_starts_only_when_it_fits_in_the_wait():
    # #285: with half of the 3.5 s wait as the limit, a 1.6 s first call was retried and the retry ran past the wait.
    from engine.final_reader import retry_fits
    assert retry_fits(1.4, 3.5) and retry_fits(1.5, 3.5)
    assert not retry_fits(1.6, 3.5) and not retry_fits(1.6, 3.0)


def test_a_driving_vehicle_must_use_the_car_template_shot():
    # Live 2026-11-15: every driving wording but the template's reversed the car, left the road or turned it around.
    from engine.final_reader import drives_without_template
    assert drives_without_template("A red car drives along a city street at night, the camera static.")
    assert not drives_without_template(car_prompt())
    assert not drives_without_template("A red car parked at the curb on the right side of the street.")
    assert not drives_without_template("A red car drives past, seen from the side.", "show me a side view")
    chat = FakeChat({"add a car driving": reply("A meadow with a red car driving across it from left to right.")})
    r = reader(chat)
    assert r("add a car driving", snap()) is None  # not used: the dictionary reads it
    assert r.stats()["verdicts"] == {"not_forward": 1}
