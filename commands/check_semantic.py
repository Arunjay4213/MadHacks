"""Real-model CPU self-check. Prepare with `python -m commands.semantic --prepare` first."""

import time

from contracts.validate import errors
from .semantic import local_reader
from .service import CommandService


def main():
    start = time.perf_counter()
    reader = local_reader()
    print(f"CPU model/index warm-up: {time.perf_counter() - start:.2f} s")
    cases = [
        ("let it pour", "rain"),
        ("give the scene a nocturnal atmosphere", "night"),
        ("make it look like the sun is coming up", "sunrise"),
        ("let precipitation pour from the sky", "rain"),
        ("add a glowing unicorn", "glowing unicorn"),
    ]
    for index, (text, label) in enumerate(cases):
        service = CommandService(semantic_reader=reader)
        start = time.perf_counter()
        result = service.handle(dict(type="final", text=text, t_client=index))
        elapsed = (time.perf_counter() - start) * 1000
        assert not result["state"]["unmatched"], text
        assert label in [change["label"] for change in result["state"]["changes"]], text
        assert not errors(2, result["steer"]), result
        print(f"PASS {text!r} -> {label} ({elapsed:.1f} ms)")
    for text in ("what is my account password", "make it nice", "don't make it freezing"):
        result = CommandService(semantic_reader=reader).handle(dict(type="final", text=text, t_client=99))
        assert result["state"]["unmatched"] and not result["state"]["changes"], text
        print(f"PASS unrelated/negated input rejected: {text!r}")
    assert reader("make it purple", {"changes": []}) is None, "purple must not become a rainbow"
    assert reader("make the weather subzero", {"changes": []}) is None, "weak matches go to the LLM"
    service = CommandService(semantic_reader=reader)
    service.handle(dict(type="final", text="add a lighthouse", t_client=100))
    result = service.handle(dict(type="final", text="add a dog and remove the lighthouse then make it nocturnal", t_client=101))
    assert [change["label"] for change in result["state"]["changes"]] == ["dog", "night"], result
    undone = service.handle(dict(type="tap", action="undo"))
    assert [change["label"] for change in undone["state"]["changes"]] == ["lighthouse"], undone
    print("PASS compound dictionary + vector edits and one-step undo")


if __name__ == "__main__":
    main()
