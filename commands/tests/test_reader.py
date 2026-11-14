import unittest

from commands.reader import KEYWORDS, read, resolve, spot


def snap(*changes):
    """changes: (id, label) pairs, using the KEYWORDS phrase and kind."""
    return {"base": "b", "can_undo": False, "changes": [
        {"id": i, "label": l, "kind": KEYWORDS[l]["kind"], "phrase": KEYWORDS[l]["phrase"], "strength": 1.0} for i, l in changes]}


class ReadTest(unittest.TestCase):
    def test_add_with_alias_and_pace(self):
        r = read("Slowly make it snow!", snap())
        self.assertEqual(r["op"], {"op": "add", "label": "winter", "kind": "season", "phrase": KEYWORDS["winter"]["phrase"], "strength": 1.0})
        self.assertEqual((r["keyword"], r["pace"]), ("snow", "slow"))
        self.assertEqual(read("suddenly night", snap())["pace"], "sudden")
        self.assertIsNone(read("add a river", snap())["pace"])

    def test_undo_reset_beat_keywords(self):
        self.assertEqual(read("go back", snap())["op"], {"op": "undo"})
        # A named undo removes the named change, not the most recent one.
        s = snap(("c1", "winter"), ("c2", "river"), ("c3", "car"))
        self.assertEqual(read("undo the winter", s)["op"], {"op": "remove", "id": "c1"})
        self.assertEqual(read("start over", snap())["op"], {"op": "reset"})

    def test_undo_synonyms_are_not_eaten_by_negation(self):
        for text in ["never mind", "Nevermind", "take that back", "revert"]:
            self.assertEqual(read(text, snap(("c1", "river")))["op"], {"op": "undo"}, text)
        self.assertIsNone(read("don't undo", snap())["op"])  # a negated command is still not read

    def test_remove_resolves_name_to_most_recent_id(self):
        s = snap(("c1", "river"), ("c2", "winter"), ("c3", "lake"))
        self.assertEqual(read("remove the stream", s)["op"], {"op": "remove", "id": "c1"})
        self.assertEqual(read("no more snow", s)["op"], {"op": "remove", "id": "c2"})
        self.assertIsNone(read("remove the dragon", s)["op"])  # nothing matches: no op, never a guess
        self.assertIsNone(read("remove the river", snap())["op"])

    def test_strength_on_existing_change_and_on_new_one(self):
        s = snap(("c1", "winter"))
        self.assertEqual(read("a lot more snow", s)["op"], {"op": "set_strength", "id": "c1", "strength": 1.3})
        self.assertEqual(read("just a little winter", s)["op"], {"op": "set_strength", "id": "c1", "strength": 0.3})
        self.assertEqual(read("more snow", s)["op"]["strength"], 1.3)
        self.assertEqual(read("less snow", s)["op"]["strength"], 0.7)
        self.assertEqual(read("a little foggy", s)["op"]["strength"], 0.3)  # new change at low strength
        self.assertEqual(read("a little foggy", s)["op"]["op"], "add")

    def test_new_scene_only_without_keyword(self):
        self.assertEqual(read("show me a beach.", snap())["op"], {"op": "new_scene", "base": "a beach"})
        self.assertEqual(read("show me winter", snap())["op"]["op"], "add")

    def test_explicit_scene_beats_keywords(self):
        self.assertEqual(read("new scene of a river in a city", snap(("c1", "winter")))["op"],
                         {"op": "new_scene", "base": "a river in a city"})

    def test_unknown_is_none(self):
        r = read("turn the tree into a dragon", snap())
        self.assertEqual(r, {"op": None, "keyword": None, "pace": None})
        self.assertIsNone(read("", snap())["op"])

    def test_whole_words_only(self):
        self.assertIsNone(read("make it win", snap())["op"])  # 'win' is not 'wind' or 'winter'
        self.assertIsNone(read("a snowboard", snap())["op"])


class ResolveTest(unittest.TestCase):
    def test_order(self):
        s = snap(("c1", "river"), ("c2", "lake"))
        self.assertEqual(resolve("c2", s), "c2")
        self.assertEqual(resolve("creek", s), "c1")
        self.assertEqual(resolve("running", s), "c1")  # phrase word
        self.assertEqual(resolve("rivr", s), "c1")  # close match
        self.assertIsNone(resolve("dragon", s))
        self.assertEqual(resolve("river", snap(("c1", "river"), ("c4", "river"))), "c4")


class SpotTest(unittest.TestCase):
    def test_spot(self):
        self.assertEqual(spot("make it wint"), None)
        self.assertEqual(spot("make it winter and"), "winter")
        self.assertIsNone(spot("remove the river"))
        self.assertIsNone(spot("go back to night"))
        self.assertIsNone(spot("new scene of a river"))


if __name__ == "__main__":
    unittest.main()
