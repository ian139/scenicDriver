"""
Mapbox Tile Source
Owner: progno-geospatial agent

Downloads satellite tiles from Mapbox API for development and training.
Production-ready implementation with caching, rate limiting, and error handling.
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Iterator
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .tile_processor import Tile

logger = logging.getLogger(__name__)


class MapboxTokenError(Exception):
    """Raised when Mapbox access token is missing or invalid."""
    pass


class MapboxDownloadError(Exception):
    """Raised when tile download fails."""
    pass


class MapboxRateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


@dataclass
class DownloadStats:
    """Statistics for a download session."""
    tiles_downloaded: int = 0
    tiles_cached: int = 0
    tiles_failed: int = 0
    bytes_downloaded: int = 0
    total_time_seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.tiles_downloaded + self.tiles_cached
        return self.tiles_cached / total if total > 0 else 0.0


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert lat/lon to tile coordinates at given zoom level.

    Uses Web Mercator (EPSG:3857) tile coordinate system.

    Args:
        lat: Latitude in degrees (-85.05 to 85.05)
        lon: Longitude in degrees (-180 to 180)
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (x, y) tile coordinates
    """
    # Clamp latitude to valid range for Web Mercator
    lat = max(-85.0511, min(85.0511, lat))

    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)

    # Clamp to valid tile range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))

    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to lat/lon (northwest corner of tile).

    Args:
        x, y: Tile coordinates
        zoom: Zoom level

    Returns:
        Tuple of (lat, lon) in degrees
    """
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def tile_to_lat_lon_center(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to lat/lon (center of tile).

    Args:
        x, y: Tile coordinates
        zoom: Zoom level

    Returns:
        Tuple of (lat, lon) for tile center
    """
    # Get corners and average
    nw_lat, nw_lon = tile_to_lat_lon(x, y, zoom)
    se_lat, se_lon = tile_to_lat_lon(x + 1, y + 1, zoom)
    return (nw_lat + se_lat) / 2, (nw_lon + se_lon) / 2




class MapboxTileSource:
    """
    Download satellite tiles from Mapbox.

    Requires MAPBOX_ACCESS_TOKEN environment variable or passed explicitly.

    Features:
    - Automatic caching to avoid re-downloading
    - Rate limiting to respect API limits
    - Retry logic for transient failures
    - Progress logging
    - Download statistics

    """

    BASE_URL = "https://api.mapbox.com/v4"

    # Default rate limit: 100 requests per second is generous for Mapbox
    DEFAULT_RATE_LIMIT = 10  # requests per second (conservative)
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.5  # exponential backoff multiplier

    def __init__(
        self,
        cache_dir: Path,
        access_token: Optional[str] = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        use_high_res: bool = True,
        style_id: str = "mapbox.satellite"
    ):
        """
        Initialize MapboxTileSource.

        Args:
            cache_dir: Directory to cache downloaded tiles
            access_token: Mapbox access token (or set MAPBOX_ACCESS_TOKEN env var)
            rate_limit: Max requests per second
            timeout: Request timeout in seconds
            use_high_res: Use @2x (512x512) tiles instead of 256x256
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.access_token = access_token or os.environ.get("MAPBOX_ACCESS_TOKEN")
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.use_high_res = use_high_res
        self.style_id = style_id

        self._last_request_time = 0.0
        self._session: Optional[requests.Session] = None
        self._stats = DownloadStats()

        # Log token status
        if self.access_token:
            # Mask token for logging
            masked = self.access_token[:8] + "..." + self.access_token[-4:]
            logger.info(f"MapboxTileSource initialized with token: {masked}")
        else:
            logger.warning(
                "MapboxTileSource: No access token found. "
                "Set MAPBOX_ACCESS_TOKEN environment variable to download tiles."
            )

    @property
    def has_token(self) -> bool:
        """Check if access token is available."""
        return self.access_token is not None

    def _ensure_token(self) -> None:
        """Raise error if token is not available."""
        if not self.access_token:
            raise MapboxTokenError(
                "Mapbox access token required. "
                "Set MAPBOX_ACCESS_TOKEN environment variable or pass access_token to constructor.\n"
                "Get a free token at: https://account.mapbox.com/access-tokens/"
            )

    def _get_session(self) -> requests.Session:
        """Get or create a requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=self.RETRY_BACKOFF,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

        return self._session

    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.rate_limit <= 0:
            return

        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _get_cache_path(self, x: int, y: int, zoom: int) -> Path:
        """Get the cache file path for a tile."""
        # Organize by zoom level for easier management
        zoom_dir = self.cache_dir / f"z{zoom}"
        zoom_dir.mkdir(parents=True, exist_ok=True)

        suffix = "@2x" if self.use_high_res else ""
        return zoom_dir / f"{x}_{y}{suffix}.png"

    def _download_tile(self, x: int, y: int, zoom: int) -> np.ndarray:
        """
        Download tile from Mapbox API.

        Args:
            x, y: Tile coordinates
            zoom: Zoom level

        Returns:
            RGB numpy array

        Raises:
            MapboxDownloadError: If download fails
            MapboxRateLimitError: If rate limited
        """
        self._ensure_token()

        suffix = "@2x" if self.use_high_res else ""
        url = (
            f"{self.BASE_URL}/{self.style_id}/{zoom}/{x}/{y}{suffix}.png"
            f"?access_token={self.access_token}"
        )

        self._rate_limit_wait()

        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout)

            if response.status_code == 401:
                raise MapboxTokenError("Invalid Mapbox access token")
            elif response.status_code == 429:
                raise MapboxRateLimitError("Mapbox rate limit exceeded. Try again later.")
            elif response.status_code == 404:
                raise MapboxDownloadError(f"Tile not found: z{zoom}/{x}/{y}")

            response.raise_for_status()

            # Track download stats
            self._stats.bytes_downloaded += len(response.content)

            # Convert to numpy array
            from io import BytesIO
            image = Image.open(BytesIO(response.content))
            return np.array(image.convert("RGB"))

        except requests.exceptions.Timeout:
            raise MapboxDownloadError(f"Timeout downloading tile z{zoom}/{x}/{y}")
        except requests.exceptions.ConnectionError as e:
            raise MapboxDownloadError(f"Connection error downloading tile z{zoom}/{x}/{y}: {e}")
        except requests.exceptions.RequestException as e:
            raise MapboxDownloadError(f"Failed to download tile z{zoom}/{x}/{y}: {e}")

    def get_tile(self, x: int, y: int, zoom: int, force_download: bool = False) -> Tile:
        """
        Fetch a single tile, using cache if available.

        Args:
            x, y: Tile coordinates
            zoom: Zoom level (typically 14-17 for satellite)
            force_download: If True, bypass cache and re-download

        Returns:
            Tile object with image data

        Raises:
            MapboxTokenError: If token is missing
            MapboxDownloadError: If download fails
        """
        cache_path = self._get_cache_path(x, y, zoom)

        if cache_path.exists() and not force_download:
            logger.debug(f"Loading from cache: {cache_path}")
            try:
                image = np.array(Image.open(cache_path).convert("RGB"))
                self._stats.tiles_cached += 1
            except Exception as exc:
                logger.warning(f"Corrupt cache file {cache_path}, re-downloading. Error: {exc}")
                force_download = True

        if not cache_path.exists() or force_download:
            logger.info(f"Downloading tile: z{zoom}/{x}/{y}")
            try:
                image = self._download_tile(x, y, zoom)
                Image.fromarray(image).save(cache_path, optimize=True)
                self._stats.tiles_downloaded += 1
                logger.debug(f"Saved to cache: {cache_path}")
            except (MapboxDownloadError, MapboxTokenError):
                self._stats.tiles_failed += 1
                raise

        lat, lon = tile_to_lat_lon_center(x, y, zoom)

        return Tile(
            image=image,
            lat=lat,
            lon=lon,
            zoom=zoom,
            x=x,
            y=y,
            source="mapbox"
        )


    def get_tiles_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        zoom: int = 15,
        max_tiles: Optional[int] = None
    ) -> Iterator[Tile]:
        """
        Get all tiles covering a bounding box.

        Args:
            min_lat, min_lon: Southwest corner
            max_lat, max_lon: Northeast corner
            zoom: Tile zoom level
            max_tiles: Maximum number of tiles to return (to prevent accidents)

        Yields:
            Tile objects

        Raises:
            ValueError: If bbox is invalid or would require too many tiles
        """
        # Validate bbox
        if min_lat >= max_lat:
            raise ValueError(f"Invalid latitude range: {min_lat} >= {max_lat}")
        if min_lon >= max_lon:
            raise ValueError(f"Invalid longitude range: {min_lon} >= {max_lon}")

        # Get tile range
        min_x, max_y = lat_lon_to_tile(min_lat, min_lon, zoom)
        max_x, min_y = lat_lon_to_tile(max_lat, max_lon, zoom)

        # Calculate total tiles
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        logger.info(f"Bbox covers {total_tiles} tiles at zoom {zoom}")

        # Safety check
        if max_tiles and total_tiles > max_tiles:
            raise ValueError(
                f"Bbox would require {total_tiles} tiles, but max_tiles={max_tiles}. "
                "Reduce bbox size or increase max_tiles."
            )

        tile_count = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if max_tiles and tile_count >= max_tiles:
                    logger.warning(f"Reached max_tiles limit ({max_tiles})")
                    return

                try:
                    yield self.get_tile(x, y, zoom)
                    tile_count += 1
                except MapboxDownloadError as e:
                    logger.error(f"Failed to get tile {x},{y}: {e}")
                    continue


    def get_stats(self) -> DownloadStats:
        """Get download statistics for this session."""
        return self._stats




    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        if self._session:
            self._session.close()
            self._session = None
        return False
