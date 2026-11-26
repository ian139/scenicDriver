"""CLI entry point for fail-closed stage-one handoff finalization."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


from scenicdriver.active_learning.finalize import main  # noqa: E402


if __name__ == "__main__":
    main()
