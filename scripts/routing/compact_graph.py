from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
from pathlib import Path

from src.route_planner.compact_graph import (
    CompactArtifactError,
    CompactArtifactInfo,
    load_compact_graph,
    validate_compact_artifact,
    write_compact_graph,
)
from src.route_planner.graph import RoadGraph


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    destination: Path
    node_count: int
    edge_count: int
    arc_count: int
    checksum_sha256: str
    source_sha256: str


def convert_json_to_compact(source: Path, destination: Path) -> ConversionResult:
    """Convert a legacy graph JSON to a validated compact artifact atomically."""
    source = Path(source)
    destination = Path(destination)
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise CompactArtifactError(f"cannot read source graph JSON {source}: {exc}") from exc
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        graph = RoadGraph.load(source)
    except Exception as exc:
        if isinstance(exc, CompactArtifactError):
            raise
        raise CompactArtifactError(f"cannot load source graph JSON {source}: {exc}") from exc
    info = write_compact_graph(graph, destination, source_sha256=source_sha256)
    return ConversionResult(
        source=source,
        destination=destination,
        node_count=info.node_count,
        edge_count=info.edge_count,
        arc_count=info.arc_count,
        checksum_sha256=info.checksum_sha256,
        source_sha256=source_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert and validate Scenic Drive compact graph artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="convert graph JSON to compact artifact")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("destination", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate a compact graph artifact")
    validate_parser.add_argument("artifact", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "convert":
            result = convert_json_to_compact(args.source, args.destination)
            print(
                f"converted {result.source} -> {result.destination}: "
                f"{result.node_count} nodes, {result.edge_count} edges, {result.arc_count} arcs, "
                f"sha256={result.checksum_sha256}"
            )
        else:
            result = validate_compact_artifact(args.artifact)
            print(
                f"valid {result.path}: {result.node_count} nodes, {result.edge_count} edges, "
                f"{result.arc_count} arcs, sha256={result.checksum_sha256}"
            )
    except CompactArtifactError as exc:
        parser.exit(2, f"compact graph error: {exc}\n")
    return 0


__all__ = [
    "CompactArtifactError",
    "CompactArtifactInfo",
    "ConversionResult",
    "convert_json_to_compact",
    "load_compact_graph",
    "validate_compact_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
