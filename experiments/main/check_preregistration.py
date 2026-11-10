#!/usr/bin/env python3
"""Check every SHA-256 hash recorded in PREREGISTRATION*.txt against the files in this folder.

  python check_preregistration.py

Every hash-locked file is byte-identical to its pre-registered version, except make_round3.py.
When this repository was reorganised, the pilot's networks moved from 04_Pilot2_Large_Networks/
and 07_API_Pilot/extra/ to ../pilot/networks/ and ../pilot/extra/, so two lines of make_round3.py
that locate them were updated. For that file, this script puts the two original lines back
in memory and checks the hash again. The file on disk is not changed.
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The only edits made to a hash-locked file: (current text, pre-registered text).
PATH_EDITS = {
    "make_round3.py": [
        (b'ROOT / "pilot" / "networks"', b'ROOT / "04_Pilot2_Large_Networks"'),
        (b'ROOT / "pilot" / "extra"', b'ROOT / "07_API_Pilot" / "extra"'),
    ],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(name, expected):
    path = HERE / name
    if not path.is_file():
        return "MISSING"
    data = path.read_bytes()
    if sha256(data) == expected:
        return "OK"
    if name in PATH_EDITS:
        for current, original in PATH_EDITS[name]:
            data = data.replace(current, original, 1)
        if sha256(data) == expected:
            return "OK (after restoring the two original input paths)"
    return "FAILED"


def main():
    failures = 0
    for record in sorted(HERE.glob("PREREGISTRATION*.txt")):
        for line in record.read_text(encoding="utf-8").splitlines():
            match = re.match(r"([0-9a-f]{64}) \*(\S+)$", line)
            if not match:
                continue
            expected, name = match.groups()
            status = check(name, expected)
            failures += status in ("FAILED", "MISSING")
            print(f"{record.name:28} {name:24} {status}")
    print("All pre-registered hashes match." if failures == 0 else f"{failures} file(s) do not match.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
