"""Reader behaviour on top of the embedded keyword dictionary (commands/keywords.json via commands/dictionary.py)."""

import unittest

from commands.reader import (KEYWORDS, MAX_CLAUSES, is_blocked, read, remove_target, resolve, split_commands, spot,
                             style_fallback)
from commands.service import CommandService
from commands.state import KINDS, SceneState


def snap(*labels):
    """A snapshot with these dictionary labels as changes c1, c2, ... in order."""
    by_label = {e["label"]: e for e in KEYWORDS.values()}
    changes = [{"id": f"c{i}", "label": l, "kind": by_label[l]["kind"], "phrase": by_label[l]["phrase"], "strength": 1.0}
               for i, l in enumerate(labels, 1)]
    return {"base": "b", "can_undo": False, "changes": changes}


def free(snapshot, cid, label, phrase):
    """Add a free-text (pass-through) change to a snapshot."""
    snapshot["changes"].append({"id": cid, "label": label, "kind": "extra", "phrase": phrase, "strength": 1.0})
    return snapshot


def op(text, snapshot=None):
    return read(text, snapshot or snap())["op"]


class NewVocabularyTest(unittest.TestCase):
    def test_day_words_add_the_day_time_change(self):
        for text in ["make it day", "Make it daytime.", "turn it to morning", "make it sunny", "make it bright", "um make it day"]:
            with self.subTest(text=text):
                o = op(text)
                self.assertEqual((o["op"], o["label"], o["kind"]), ("add", "day", "time"))

    def test_day_replaces_night_in_state(self):
        state = SceneState()
        state.apply(op("make it night"))
        state.apply(op("make it sunny", state.snapshot()))
        self.assertEqual([c["label"] for c in state.snapshot()["changes"]], ["day"])

    def test_demo_objects(self):
        for text, label in [("add a car", "car"), ("add a dog", "dog"), ("build a snowman", "snowman"),
                            ("add some people", "people"), ("add a castle", "castle")]:
            with self.subTest(text=text):
                o = op(text)
                self.assertEqual((o["op"], o["label"], o["kind"]), ("add", label, "extra"))


class NormalizationTest(unittest.TestCase):
    def test_case_punctuation_and_unicode_space(self):
        for text in ["MAKE IT WINTER!!!", "make it winter", "make it\twinter\n", "make it winter…"]:
            self.assertEqual(op(text)["label"], "winter", repr(text))

    def test_hyphens_and_dashes(self):
        self.assertEqual(op("make it night-time")["label"], "night")
        self.assertEqual(op("make it night—slowly")["label"], "night")
        self.assertEqual(read("make it night—slowly", snap())["pace"], "slow")

    def test_longest_alias_wins(self):
        self.assertEqual(op("add a snowman")["label"], "snowman")
        self.assertEqual(op("add a snow man")["label"], "snowman")
        self.assertEqual(op("add a snow-capped mountain")["label"], "mountains")

    def test_whole_words_only(self):
        for text in ["a snowboard", "make it win", "a carpet", "a dogma", "a downdraft"]:
            o = op(text)
            self.assertTrue(o is None or o["label"] not in ("winter", "wind", "car", "dog"), text)


class DisambiguationTest(unittest.TestCase):
    def test_object_noun_beats_adjective_aliases(self):
        for text, label in [("add a bright red car", "car"), ("add dark clouds", "clouds"), ("add lightning bugs", "fireflies"),
                            ("add a snow man", "snowman"), ("add a dark forest", "trees"), ("add a snowy mountain", "mountains")]:
            with self.subTest(text=text):
                self.assertEqual(op(text)["label"], label)

    def test_adjective_alone_still_works(self):
        self.assertEqual(op("make it bright")["label"], "day")
        self.assertEqual(op("make it dark")["label"], "night")
        self.assertEqual(op("make it snowy")["label"], "winter")

    def test_alias_words_do_not_leak_into_strength(self):
        # "light" in "light house" is part of the noun, not "a light amount".
        self.assertEqual(op("add a light house")["strength"], 1.0)
        self.assertEqual(op("add a little rain")["strength"], 0.3)


class SafeTargetingTest(unittest.TestCase):
    def test_absent_recognized_noun_never_falls_through_to_adjectives(self):
        # autumn's phrase says "red", sunset's says "golden": neither is a car.
        self.assertIsNone(op("remove the red car", snap("autumn", "sunset")))
        self.assertIsNone(resolve("the red car", snap("autumn", "sunset")))
        self.assertIsNone(op("remove the red car", snap("autumn")))
        self.assertIsNone(op("remove the golden thing", snap("sunset")))  # colour words never target a change

    def test_recognized_noun_targets_its_change(self):
        s = snap("autumn", "car")
        self.assertEqual(op("remove the red car", s), {"op": "remove", "id": "c2"})
        self.assertEqual(op("remove the snow man", snap("winter", "snowman")), {"op": "remove", "id": "c2"})
        self.assertEqual(op("get rid of the snowy mountains", snap("winter", "mountains")), {"op": "remove", "id": "c2"})
        self.assertEqual(op("remove the water fall", snap("river", "waterfall")), {"op": "remove", "id": "c2"})

    def test_multiword_alias_beats_single_word(self):
        # "snow capped mountains" names mountains, not winter.
        self.assertEqual(resolve("the snow capped mountains", snap("winter", "mountains")), "c2")

    def test_existing_fallbacks_kept(self):
        s = snap("river", "lake")
        self.assertEqual(resolve("running", s), "c1")  # phrase word, no dictionary noun in the query
        self.assertEqual(resolve("rivr", s), "c1")  # close noun name

    def test_fuzzy_ignores_stopwords_and_short_words(self):
        s = snap("river", "lake")
        for q in ["the", "it", "that", "a", "then", "some"]:
            self.assertIsNone(resolve(q, s), q)

    def test_free_label_is_removable(self):
        s = free(snap("winter"), "c2", "lion", "a lion")
        self.assertEqual(op("remove the lion", s), {"op": "remove", "id": "c2"})
        # A free label containing an alias word is matched as a whole phrase, before the alias rule.
        s = free(snap("night"), "c2", "dark knight", "a dark knight")
        self.assertEqual(op("remove the dark knight", s), {"op": "remove", "id": "c2"})
        self.assertEqual(op("remove the night", s), {"op": "remove", "id": "c1"})
        self.assertIsNone(op("remove the dark knight", snap("night")))  # knight absent: never removes the night


class PassThroughTest(unittest.TestCase):
    def test_unknown_object_passes_through(self):
        self.assertEqual(op("add a lion"), {"op": "add", "label": "lion", "kind": "extra", "phrase": "a lion", "strength": 1.0})
        self.assertEqual(op("Add a big red balloon, please.")["label"], "big red balloon")
        o = op("put some hot air balloons in the sky")
        self.assertEqual((o["label"], o["phrase"]), ("hot air balloons", "some hot air balloons in the sky"))

    def test_unknown_direct_object_beats_known_words_in_its_location(self):
        o = op("add a lion near the river")
        self.assertEqual((o["label"], o["phrase"]), ("lion", "a lion near the river"))
        o = op("i want a dragon flying over the mountains")
        self.assertEqual((o["label"], o["phrase"]), ("dragon", "a dragon flying over the mountains"))
        self.assertEqual(op("add a tall building")["label"], "tall building")  # -ing nouns are not cut
        # An unknown noun beats an adjective-like alias in front of it.
        o = op("add a bright unicorn")
        self.assertEqual((o["label"], o["phrase"]), ("bright unicorn", "a bright unicorn"))
        self.assertEqual(op("add a dark knight")["label"], "dark knight")
        self.assertEqual(op("add a snowy owl")["label"], "snowy owl")
        # ...but when nothing real is left after the alias and modifier words, the alias is the meaning.
        for text, label in [("add a little rain", "rain"), ("add a light breeze", "wind"), ("add a heavy storm", "storm"),
                            ("add some frost", "winter"), ("add a stormy sky", "storm"), ("add a bright red car", "car")]:
            self.assertEqual(op(text)["label"], label, text)
        # A known direct object still wins: the location does not matter.
        self.assertEqual(op("put a boat on the lake")["label"], "boat")
        self.assertEqual(op("add more snow")["label"], "winter")
        self.assertEqual(op("make it winter and add a dog")["label"], "dog")

    def test_pass_through_never_guessed_early_and_never_for_negation(self):
        self.assertIsNone(spot("add a lion"))
        self.assertIsNone(op("don't add a lion"))
        self.assertIsNone(op("do not  add a lion"))  # repeated spaces
        self.assertIsNone(op("don’t add a lion"))

    def test_pass_through_change_is_valid_state(self):
        state = SceneState()
        self.assertTrue(state.apply(op("add a lion")))
        self.assertIn(state.snapshot()["changes"][0]["kind"], KINDS)

    def test_empty_or_filler_objects_are_not_added(self):
        for text in ["add", "add a", "add some please", "add it", "put the whole thing on a cliff edge", "add something big"]:
            self.assertIsNone(op(text), text)


class UnsupportedStaysUnmatchedTest(unittest.TestCase):
    def test_negation_and_transforms(self):
        for text in ["don't make it winter", "do not   add a river", "don’t make it night", "never snow",
                     "turn the tree into a lighthouse", "replace the river with a road", "swap the dog for a cat"]:
            with self.subTest(text=text):
                self.assertIsNone(op(text, snap("river", "dog")))
                self.assertIsNone(spot(text))

    def test_is_blocked_marks_deliberate_misses_only(self):
        for text in ["don't make it winter", "do not add a lion", "don\u2019t make it night", "turn the tree into a lighthouse"]:
            self.assertTrue(is_blocked(text), text)
        for text in ["make it winter", "never mind", "it feels too cheerful, make it moody", "add a lion"]:
            self.assertFalse(is_blocked(text), text)

    def test_never_mind_is_still_undo(self):
        self.assertEqual(op("never mind"), {"op": "undo"})
        self.assertEqual(op("Never mind!"), {"op": "undo"})

    def test_no_prefix_guesses(self):
        for partial in ["make it wint", "add a do", "make it ni", "add a snowm"]:
            self.assertIsNone(spot(partial), partial)


class SplitCommandsTest(unittest.TestCase):
    def test_root_example(self):
        self.assertEqual(split_commands("add a dog and remove the lighthouse"), ["add a dog", "remove the lighthouse"])

    def test_separators_need_a_new_action(self):
        self.assertEqual(split_commands("Make it winter; add a river."), ["make it winter", "add a river"])
        self.assertEqual(split_commands("make it winter, then add a river"), ["make it winter", "add a river"])
        self.assertEqual(split_commands("go back and then make it night"), ["go back", "make it night"])
        self.assertEqual(split_commands("make it winter, also add a river"), ["make it winter", "also add a river"])
        self.assertEqual(split_commands("Make it winter. Add a river!"), ["make it winter", "add a river"])

    def test_compound_descriptions_are_kept(self):
        for text in ["make it black and white", "add a black and white dog", "play some rock and roll",
                     "add red and orange leaves", "add a lion and a tiger"]:
            self.assertEqual(split_commands(text), [normalize_for_test(text)], text)

    def test_repeated_verb_for_recognized_noun_lists(self):
        self.assertEqual(split_commands("add a dog and a cat"), ["add a dog", "add a cat"])
        self.assertEqual(split_commands("remove river and lighthouse"), ["remove river", "remove lighthouse"])
        self.assertEqual(split_commands("make it winter and night"), ["make it winter", "make it night"])
        self.assertEqual(split_commands("get rid of the river, the lighthouse and the dog"),
                         ["get rid of the river", "get rid of the lighthouse", "get rid of the dog"])

    def test_aliases_and_each_clause_reads(self):
        clauses = split_commands("add a puppy and remove the light house")
        self.assertEqual(clauses, ["add a puppy", "remove the light house"])
        self.assertEqual(op(clauses[0])["label"], "dog")
        self.assertEqual(op(clauses[1], snap("lighthouse")), {"op": "remove", "id": "c1"})

    def test_sequential_reference_in_trial_state(self):
        clauses = split_commands("add a dog and remove the dog")
        self.assertEqual(clauses, ["add a dog", "remove the dog"])
        state = SceneState()
        for clause in clauses:
            self.assertTrue(state.apply(read(clause, state.snapshot())["op"]))
        self.assertEqual(state.snapshot()["changes"], [])

    def test_negated_clause_is_kept_separate_and_unmatched(self):
        clauses = split_commands("don't add a dog and remove the lighthouse")
        self.assertEqual(clauses, ["don't add a dog", "remove the lighthouse"])
        self.assertIsNone(op(clauses[0]))
        self.assertTrue(is_blocked(clauses[0]))

    def test_full_stops_and_ellipses_only_split_before_an_action(self):
        self.assertEqual(split_commands("Make it... winter?"), ["make it winter"])
        self.assertEqual(split_commands("Make it\u2026 winter"), ["make it winter"])
        self.assertEqual(split_commands("add a St. Bernard"), ["add a st bernard"])
        self.assertEqual(split_commands("Make it winter. Add a river."), ["make it winter", "add a river"])

    def test_fullwidth_separators(self):
        self.assertEqual(split_commands("make it winter\uff1b add a river"), ["make it winter", "add a river"])
        self.assertEqual(split_commands("make it winter\uff01add a river\uff1f"), ["make it winter", "add a river"])

    def test_single_and_empty(self):
        self.assertEqual(split_commands("make it winter"), ["make it winter"])
        self.assertEqual(split_commands(""), [])
        self.assertEqual(split_commands(" ; , . "), [])

    def test_clause_limit_is_exported(self):
        self.assertEqual(MAX_CLAUSES, 16)
        self.assertEqual(len(split_commands(" and ".join(["undo"] * 20))), 20)  # the caller rejects > MAX_CLAUSES


def normalize_for_test(text):
    from commands.dictionary import normalize
    return normalize(text)


class RemoveIntentTest(unittest.TestCase):
    def test_remove_target_matches_every_remove_verb(self):
        for text, target in [("remove the river", "the river"), ("without the river", "the river"), ("drop the dog", "dog"),
                             ("lose the clouds", "clouds"), ("stop the rain", "rain"), ("no more snow", "snow"),
                             ("get rid of the boat please", "the boat please")]:
            self.assertEqual(remove_target(text), target, text)
        for text in ["add a river", "make it winter", "a raindrop", ""]:
            self.assertIsNone(remove_target(text), text)


class StyleFallbackTest(unittest.TestCase):
    def test_moods_and_looks_become_a_style_change(self):
        self.assertEqual(style_fallback("make it spooky"),
                         {"op": "add", "label": "spooky", "kind": "extra", "phrase": "the whole scene looks spooky", "strength": 1.0})
        self.assertEqual(style_fallback("make it look like the 1920s")["phrase"], "the whole scene looks like the 1920s")
        self.assertEqual(style_fallback("make it feel gloomy")["label"], "gloomy")
        self.assertEqual(style_fallback("make it black and white")["label"], "black and white")
        self.assertEqual(style_fallback("make everything purple")["label"], "purple")

    def test_never_for_negation_empty_or_long_or_known_words(self):
        for text in ["make it nice", "make it better", "make it look good", "make it really nice",
                     "don't make it spooky", "make it", "make it really really very extremely strange and odd today",
                     "make it winter", "add a dog", "remove the river", "it"]:
            self.assertIsNone(style_fallback(text), text)


class CompatibilityTest(unittest.TestCase):
    def test_keywords_shape_and_kinds(self):
        for alias, entry in KEYWORDS.items():
            self.assertEqual(set(entry), {"label", "kind", "phrase"}, alias)
            self.assertIn(entry["kind"], KINDS, alias)

    def test_service_end_to_end_with_new_keyword(self):
        svc = CommandService()
        r = svc.handle({"type": "final", "text": "make it sunny", "t_client": 1})
        self.assertEqual([c["label"] for c in r["state"]["changes"]], ["day"])
        self.assertIsNotNone(r["steer"])
        r = svc.handle({"type": "final", "text": "add a lion", "t_client": 2})
        self.assertIn("a lion", r["steer"]["prompt"])
        self.assertTrue(any("castle" in p for p in svc.likely_prompts()))


if __name__ == "__main__":
    unittest.main()
