"""Check every contracts/examples/*.json against its schema in contracts/schemas/.

An example file named "<n>-....json" is checked against "schemas/<n>-*.schema.json".
Run from anywhere: python contracts/validate.py  (needs jsonschema)
Exits non-zero if any example is invalid or has no schema.
Other code can import `errors(contract_number, message)` to check a single message.
"""

import json
import sys
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent


@cache
def validator(contract: int) -> Draft202012Validator:
    matches = sorted((HERE / "schemas").glob(f"{contract}-*.schema.json"))
    if len(matches) != 1:
        raise LookupError(f"expected one schema for contract {contract}, found {len(matches)}")
    schema = json.loads(matches[0].read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def errors(contract: int, message: object) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in validator(contract).iter_errors(message)]


def main() -> int:
    failed = 0
    examples = sorted((HERE / "examples").glob("*.json"))
    for path in examples:
        contract = int(path.name.split("-", 1)[0])
        try:
            problems = errors(contract, json.loads(path.read_text()))
        except (LookupError, json.JSONDecodeError) as exc:
            problems = [str(exc)]
        status = "ok" if not problems else "FAIL"
        print(f"{status:4} contract {contract}  {path.name}")
        for p in problems:
            print(f"       {p}")
        failed += bool(problems)
    print(f"{len(examples) - failed}/{len(examples)} examples valid")
    return 1 if failed or not examples else 0


if __name__ == "__main__":
    sys.exit(main())
