"""
Download Mapbox tiles for a bounding box.

Examples:
  python scripts/ingest/download_bbox_tiles.py \
    --min-lat 40.018 --min-lon -75.2284 \
    --max-lat 40.0734 --max-lon -75.185 \
    --zoom 16 --style mapbox.satellite \
    --output data/raw/images/satellite

  python scripts/ingest/download_bbox_tiles.py \
    --min-lat 40.018 --min-lon -75.2284 \
    --max-lat 40.0734 --max-lon -75.185 \
    --zoom 16 --style mapbox.terrain-rgb \
    --output data/raw/images/terrain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.mapbox import MapboxTileSource, lat_lon_to_tile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mapbox tiles for bbox")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--style", type=str, default="mapbox.satellite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--rate-limit", type=float, default=10.0)
    parser.add_argument("--high-res", action="store_true")
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--s3-prefix", type=str, default=None)
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--s3-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    source = MapboxTileSource(
        cache_dir=output_dir,
        rate_limit=args.rate_limit,
        use_high_res=args.high_res,
        style_id=args.style,
    )

    if args.s3_only and not args.s3_bucket:
        raise ValueError("--s3-only requires --s3-bucket")

    if args.s3_only:
        tiles = _download_tiles_to_s3_only(
            source=source,
            min_lat=args.min_lat,
            min_lon=args.min_lon,
            max_lat=args.max_lat,
            max_lon=args.max_lon,
            zoom=args.zoom,
            max_tiles=args.max_tiles,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            style=args.style,
            output_dir=output_dir,
        )
    else:
        tiles = list(
            source.get_tiles_for_bbox(
                min_lat=args.min_lat,
                min_lon=args.min_lon,
                max_lat=args.max_lat,
                max_lon=args.max_lon,
                zoom=args.zoom,
                max_tiles=args.max_tiles,
            )
        )

    if args.s3_bucket and not args.s3_only:
        _upload_tiles_to_s3(
            tiles=tiles,
            output_dir=output_dir,
            zoom=args.zoom,
            high_res=args.high_res,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            delete_local=args.delete_local,
            style=args.style,
        )

    stats = source.get_stats()
    if args.s3_only:
        key_prefix = _resolve_s3_prefix(
            prefix=args.s3_prefix,
            style=args.style,
            output_dir=output_dir,
            zoom=args.zoom,
        )
        print(
            f"Downloaded {len(tiles)} tiles and uploaded to s3://{args.s3_bucket}/{key_prefix}/ "
            f"(downloaded={stats.tiles_downloaded}, failed={stats.tiles_failed})"
        )
    else:
        print(
            f"Saved {len(tiles)} tiles to {output_dir} "
            f"(downloaded={stats.tiles_downloaded}, cached={stats.tiles_cached}, failed={stats.tiles_failed})"
        )


def _default_s3_prefix(style: str) -> str:
    if "terrain" in style:
        return "raw/images/terrain"
    return "raw/images/satellite"


def _resolve_s3_prefix(*, prefix: str | None, style: str, output_dir: Path, zoom: int) -> str:
    if prefix is None:
        raw_prefix = _default_s3_prefix(style)
    else:
        raw_prefix = prefix.strip().strip("/")
        if not (
            raw_prefix in {"raw/images/satellite", "raw/images/terrain"}
            or raw_prefix.startswith(("raw/images/satellite/", "raw/images/terrain/"))
        ):
            raise ValueError(
                f"Invalid S3 prefix '{prefix}': must be canonical "
                "'raw/images/{satellite,terrain}' form"
            )
    base_prefix = raw_prefix
    region = None
    zoom_dir = f"z{zoom}"
    # Detect layout .../z14/<region> from local output path and preserve it in S3 keys.
    if output_dir.parent.name == zoom_dir and output_dir.name != zoom_dir:
        region = output_dir.name
    if region:
        return f"{base_prefix}/{zoom_dir}/{region}"
    return f"{base_prefix}/{zoom_dir}"


def _upload_tiles_to_s3(
    *,
    tiles: list,
    output_dir: Path,
    zoom: int,
    high_res: bool,
    bucket: str,
    prefix: str | None,
    delete_local: bool,
    style: str,
) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 uploads. Run: uv sync") from exc

    s3 = boto3.client("s3")
    suffix = "@2x" if high_res else ""
    key_prefix = _resolve_s3_prefix(prefix=prefix, style=style, output_dir=output_dir, zoom=zoom)

    uploaded = 0
    for tile in tiles:
        local_path = output_dir / f"z{zoom}" / f"{tile.x}_{tile.y}{suffix}.png"
        if not local_path.exists():
            continue
        key = f"{key_prefix}/{tile.x}_{tile.y}{suffix}.png"
        s3.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "image/png"},
        )
        uploaded += 1
        if delete_local:
            local_path.unlink(missing_ok=True)

    print(f"Uploaded {uploaded} tiles to s3://{bucket}/{key_prefix}/")


def _download_tiles_to_s3_only(
    *,
    source: MapboxTileSource,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    zoom: int,
    max_tiles: int | None,
    bucket: str,
    prefix: str | None,
    style: str,
    output_dir: Path,
) -> list:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 uploads. Run: uv sync") from exc

    if min_lat >= max_lat:
        raise ValueError(f"Invalid latitude range: {min_lat} >= {max_lat}")
    if min_lon >= max_lon:
        raise ValueError(f"Invalid longitude range: {min_lon} >= {max_lon}")

    # Get tile range
    min_x, max_y = lat_lon_to_tile(min_lat, min_lon, zoom)
    max_x, min_y = lat_lon_to_tile(max_lat, max_lon, zoom)
    total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    if max_tiles and total_tiles > max_tiles:
        raise ValueError(
            f"Bbox would require {total_tiles} tiles, but max_tiles={max_tiles}. "
            "Reduce bbox size or increase max_tiles."
        )

    from io import BytesIO
    from PIL import Image

    s3 = boto3.client("s3")
    key_prefix = _resolve_s3_prefix(prefix=prefix, style=style, output_dir=output_dir, zoom=zoom)
    tiles = []
    tile_count = 0

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if max_tiles and tile_count >= max_tiles:
                return tiles
            try:
                image = source._download_tile(x, y, zoom)
                tile_count += 1
            except Exception:
                source.get_stats().tiles_failed += 1
                continue

            buf = BytesIO()
            Image.fromarray(image).save(buf, format="PNG", optimize=True)
            buf.seek(0)
            key = f"{key_prefix}/{x}_{y}.png"
            s3.upload_fileobj(buf, bucket, key, ExtraArgs={"ContentType": "image/png"})
            tiles.append((x, y))
            source.get_stats().tiles_downloaded += 1

    return tiles


if __name__ == "__main__":
    main()
