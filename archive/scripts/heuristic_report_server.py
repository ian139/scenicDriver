"""
Serve a heuristic report directory with a lightweight local server.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heuristics.labeler import HeuristicLabelerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a heuristic report")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-open", dest="open", action="store_false")
    parser.set_defaults(open=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.path is None and args.run_name is None:
        raise ValueError("Provide --run-name or --path")

    if args.path:
        report_dir = Path(args.path)
    else:
        cfg = HeuristicLabelerConfig()
        report_dir = Path(cfg.processed_dir) / "heuristic_runs" / args.run_name / "report"

    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(report_dir), **handler_kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/index.html"
    if args.open:
        webbrowser.open(url)
    print(f"Serving report at {url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
