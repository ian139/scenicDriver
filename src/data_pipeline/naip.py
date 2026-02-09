"""
NAIP Data Access
Owner: progno-geospatial agent

Downloads and processes NAIP (National Agriculture Imagery Program) data
from Amazon S3 public bucket.

NAIP provides 1-meter resolution aerial imagery for the contiguous US.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, Optional
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import requests

logger = logging.getLogger(__name__)


@dataclass
class NAIPTile:
    """NAIP tile metadata."""
    state: str  # Two-letter state code
    year: int  # Imagery year
    resolution: str  # e.g., '100cm', '60cm'
    quadrangle: str  # USGS quadrangle name
    filename: str
    s3_key: str

    @property
    def full_s3_path(self) -> str:
        return f"s3://naip-visualization/{self.s3_key}"


class NAIPDownloader:
    """
    Download NAIP imagery from AWS S3.

    NAIP bucket: s3://naip-visualization (requester pays)

    Structure:
        naip-visualization/
            {state}/
                {year}/
                    {resolution}/
                        rgbir/
                            {quadrangle}/
                                {filename}.tif
    """

    BUCKET = "naip-visualization"

    def __init__(
        self,
        cache_dir: Path,
        aws_profile: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.aws_profile = aws_profile

        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        config = Config(retries={"max_attempts": 5, "mode": "standard"})
        self.s3 = session.client("s3", config=config)

    def list_available_states(self, year: int = 2021) -> List[str]:
        """List states with available NAIP imagery for a year."""
        paginator = self.s3.get_paginator("list_objects_v2")
        prefixes = []
        try:
            for page in paginator.paginate(Bucket=self.BUCKET, Delimiter="/"):
                for prefix in page.get("CommonPrefixes", []):
                    name = prefix.get("Prefix", "").strip("/")
                    if name:
                        prefixes.append(name)
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to list NAIP states: {exc}") from exc

        if year is None:
            return sorted(prefixes)

        available = []
        for state in prefixes:
            try:
                resp = self.s3.list_objects_v2(
                    Bucket=self.BUCKET,
                    Prefix=f"{state}/{year}/",
                    MaxKeys=1,
                    RequestPayer="requester",
                )
                if resp.get("KeyCount", 0) > 0:
                    available.append(state)
            except (BotoCoreError, ClientError) as exc:
                logger.warning("Failed to probe state %s: %s", state, exc)
        return sorted(available)

    def list_tiles_for_state(
        self,
        state: str,
        year: int = 2021
    ) -> Iterator[NAIPTile]:
        """
        List all available tiles for a state.

        Args:
            state: Two-letter state code (e.g., 'CA', 'TX')
            year: Imagery year

        Yields:
            NAIPTile objects
        """
        state = state.upper()
        prefix = f"{state}/{year}/"
        paginator = self.s3.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(
                Bucket=self.BUCKET,
                Prefix=prefix,
                RequestPayer="requester",
            ):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not key or not key.endswith(".tif"):
                        continue
                    parts = key.split("/")
                    if len(parts) < 5:
                        continue
                    # {state}/{year}/{resolution}/rgbir/{quadrangle}/{filename}.tif
                    resolution = parts[2]
                    quadrangle = parts[-2]
                    filename = parts[-1]
                    yield NAIPTile(
                        state=state,
                        year=int(parts[1]),
                        resolution=resolution,
                        quadrangle=quadrangle,
                        filename=filename,
                        s3_key=key,
                    )
        except (BotoCoreError, ClientError, NoCredentialsError) as exc:
            raise RuntimeError(f"Failed to list NAIP tiles: {exc}") from exc

    def download_tile(self, tile: NAIPTile) -> Path:
        """
        Download a NAIP tile.

        Args:
            tile: NAIPTile to download

        Returns:
            Path to downloaded file
        """
        output_path = self.cache_dir / tile.state / str(tile.year) / tile.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            logger.info(f"Tile already cached: {output_path}")
            return output_path

        try:
            self.s3.download_file(
                self.BUCKET,
                tile.s3_key,
                str(output_path),
                ExtraArgs={"RequestPayer": "requester"},
            )
        except (BotoCoreError, ClientError, NoCredentialsError) as exc:
            raise RuntimeError(f"Failed to download NAIP tile: {exc}") from exc
        return output_path

    def get_tiles_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        year: int = 2021
    ) -> List[NAIPTile]:
        """
        Find NAIP tiles that cover a bounding box.

        Args:
            min_lat, min_lon, max_lat, max_lon: Bounding box
            year: Imagery year

        Returns:
            List of tiles covering the area
        """
        stac_url = os.environ.get(
            "NAIP_STAC_URL",
            "https://planetarycomputer.microsoft.com/api/stac/v1",
        )
        collection = os.environ.get("NAIP_STAC_COLLECTION", "naip")
        search_url = f"{stac_url.rstrip('/')}/search"
        datetime = f"{year}-01-01/{year}-12-31"

        payload = {
            "collections": [collection],
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "datetime": datetime,
            "limit": 100,
        }

        try:
            response = requests.post(search_url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"STAC query failed: {exc}. "
                "Set NAIP_STAC_URL/NAIP_STAC_COLLECTION or use list_tiles_for_state."
            ) from exc

        data = response.json()
        features = data.get("features", [])
        tiles: list[NAIPTile] = []

        for item in features:
            props = item.get("properties", {})
            item_year = int(str(props.get("year", year))[:4])
            state = str(props.get("naip:state", props.get("state", "NA"))).upper()
            resolution = str(props.get("naip:resolution", props.get("resolution", "unknown")))
            quadrangle = str(props.get("naip:quad_id", props.get("quad_id", "unknown")))

            assets = item.get("assets", {})
            href = None
            for key in ("image", "data", "visual", "rendered_preview"):
                if key in assets:
                    href = assets[key].get("href")
                    if href:
                        break
            if href is None:
                continue

            s3_key = _extract_naip_s3_key(href)
            if s3_key is None:
                continue

            filename = s3_key.split("/")[-1]
            tiles.append(
                NAIPTile(
                    state=state,
                    year=item_year,
                    resolution=resolution,
                    quadrangle=quadrangle,
                    filename=filename,
                    s3_key=s3_key,
                )
            )

        if not tiles:
            raise RuntimeError(
                "No NAIP tiles found for bbox. "
                "Try a different year or check STAC endpoint settings."
            )

        return tiles

    def download_region(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        year: int = 2021,
        max_tiles: Optional[int] = None
    ) -> Iterator[Path]:
        """
        Download all tiles for a region.

        Args:
            Bounding box coordinates
            year: Imagery year
            max_tiles: Maximum number of tiles to download

        Yields:
            Paths to downloaded tiles
        """
        tiles = self.get_tiles_for_bbox(min_lat, min_lon, max_lat, max_lon, year)

        if max_tiles:
            tiles = tiles[:max_tiles]

        for tile in tiles:
            yield self.download_tile(tile)


def _extract_naip_s3_key(href: str) -> Optional[str]:
    """Extract an S3 key from a NAIP asset href if possible."""
    if href.startswith("s3://naip-visualization/"):
        return href.replace("s3://naip-visualization/", "", 1)
    if "naip-visualization.s3.amazonaws.com/" in href:
        return href.split("naip-visualization.s3.amazonaws.com/")[-1]
    return None
