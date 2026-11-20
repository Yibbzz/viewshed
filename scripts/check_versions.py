"""Check that every place the version is written agrees, and with the release tag.

    python scripts/check_versions.py            # the files agree with each other
    python scripts/check_versions.py v0.0.2     # ... and with this tag

Run by both CI workflows; the wheels workflow passes the tag, and publishing
waits for it. Standard library only (no TOML parser needed for these files).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACES = {
    "crates/viewshed/Cargo.toml": r'(?m)^version\s*=\s*"([^"]+)"',
    "crates/viewshed-py/Cargo.toml": r'(?m)^version\s*=\s*"([^"]+)"',
    "python/pyproject.toml": r'(?m)^version\s*=\s*"([^"]+)"',
    "python/viewshed/__init__.py": r'(?m)^__version__\s*=\s*"([^"]+)"',
}


def versions():
    found = {}
    for path, pattern in PLACES.items():
        m = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
        found[path] = m.group(1) if m else None
    return found


def main(argv):
    found = versions()
    for path, v in found.items():
        print(f"{v or 'MISSING':>10}  {path}")
    distinct = set(found.values())
    errors = []
    if None in distinct:
        errors.append("a version is missing")
    if len(distinct) > 1:
        errors.append("the versions differ")
    if len(argv) > 1 and argv[1]:
        tag = argv[1].removeprefix("refs/tags/")
        if tag != f"v{next(iter(distinct))}" or len(distinct) > 1:
            errors.append(f"tag {tag} does not match the version")
    for e in errors:
        print(f"error: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
