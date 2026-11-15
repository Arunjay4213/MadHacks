"""engine/README.md's live command line is the service's ExecStart, word for word (Greptile on PRs #101 and #125)."""

import re
import shlex
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def test_readme_live_command_is_the_service_exec_start():
    service = (ENGINE / "deploy" / "madhacks-engine.service").read_text()
    exec_start = shlex.split(re.search(r"^ExecStart=(.+)$", service, re.M).group(1))
    readme = (ENGINE / "README.md").read_text()
    block = re.search(r"The live command line is `ExecStart`.*?\n```[^\n]*\n(.*?)```", readme, re.S)
    assert block, "README lost its live command block"
    documented = shlex.split(block.group(1).replace("\\\n", " "))
    assert Path(exec_start[0]).name == documented[0] == "python"
    assert exec_start[1:] == documented[1:]
