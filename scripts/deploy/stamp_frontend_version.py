"""Stamp or verify the content-derived cache version of the hosted viewer."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys


ASSETS = ("viewer.js", "styles.css")
DEFAULT_APP_DIR = Path(__file__).resolve().parents[2] / "apps/new_england_north"


def stamp_frontend(app_dir: Path, *, check: bool = False) -> str:
    digest = hashlib.sha256()
    for asset in ASSETS:
        digest.update((app_dir / asset).read_bytes())
    version = digest.hexdigest()
    index_path = app_dir / "index.html"
    original = index_path.read_text(encoding="utf-8")
    stamped = original
    for asset in ASSETS:
        pattern = re.compile(
            rf"(\b(?:src|href)=[\"']/{re.escape(asset)})(?:\?v=[^\"']*)?([\"'])"
        )
        stamped, count = pattern.subn(
            lambda match: f"{match[1]}?v={version}{match[2]}", stamped
        )
        if count != 1:
            raise ValueError(f"Expected exactly one /{asset} reference in {index_path}; found {count}")
    if check:
        if original != stamped:
            raise ValueError("Frontend stamp does not match viewer.js + styles.css; re-stamp before deployment")
    elif original != stamped:
        index_path.write_text(stamped, encoding="utf-8")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--check", action="store_true", help="Fail if index.html is not stamped for the current assets")
    args = parser.parse_args()
    try:
        version = stamp_frontend(args.app_dir, check=args.check)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{'Verified' if args.check else 'Stamped'} frontend version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
