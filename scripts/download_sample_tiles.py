#!/usr/bin/env python3
"""
Download Sample Tiles Script
Owner: progno-geospatial agent

Downloads satellite tiles for scenic locations to use as training/test data.

Usage:
    python scripts/download_sample_tiles.py

    # With custom output directory
    python scripts/download_sample_tiles.py --output data/tiles/scenic

    # Higher zoom level for more detail
    python scripts/download_sample_tiles.py --zoom 16

    # Include surrounding tiles for each location
    python scripts/download_sample_tiles.py --surrounding 1

Environment:
    MAPBOX_ACCESS_TOKEN - Required for downloading tiles from Mapbox API
                          Get a free token at: https://account.mapbox.com/access-tokens/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.mapbox import (
    MapboxTileSource,
    MapboxTokenError,
    MapboxDownloadError,
    lat_lon_to_tile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Scenic locations across the US - diverse terrain types for training
SCENIC_LOCATIONS = [
    # National Parks - dramatic landscapes
    (37.7456, -119.5936, "Yosemite Valley, CA"),
    (36.1070, -112.1130, "Grand Canyon South Rim, AZ"),
    (36.2388, -121.7873, "Big Sur, CA"),
    (44.4280, -110.5885, "Old Faithful, Yellowstone, WY"),
    (36.5058, -118.5754, "Sequoia National Park, CA"),
    (37.2189, -113.1850, "Zion Canyon, UT"),
    (38.7331, -109.5925, "Arches National Park, UT"),
    (48.7596, -113.7870, "Glacier National Park, MT"),

    # Coastal scenes
    (32.8972, -117.2569, "La Jolla Cove, CA"),
    (34.0259, -118.7798, "Malibu Coast, CA"),
    (45.8788, -123.9619, "Cannon Beach, OR"),
    (41.9185, -124.1660, "Redwood Coast, CA"),
    (25.7617, -80.1918, "Miami Beach, FL"),
    (21.2770, -157.8266, "Waikiki Beach, HI"),

    # Mountain roads - the core use case
    (39.3788, -105.8786, "Loveland Pass, CO"),
    (39.1178, -106.4453, "Independence Pass, CO"),
    (37.8107, -119.8756, "Tioga Pass, CA"),
    (48.7340, -121.7125, "North Cascades Highway, WA"),
    (35.6085, -83.4252, "Blue Ridge Parkway, NC"),
    (44.4499, -73.2342, "Green Mountains, VT"),

    # Desert landscapes
    (36.2442, -116.8234, "Death Valley, CA"),
    (32.3795, -110.9378, "Saguaro National Park, AZ"),
    (31.9231, -109.8735, "Chiricahua Mountains, AZ"),
    (35.0267, -111.0225, "Sunset Crater, AZ"),

    # Lakes and water features
    (39.0968, -120.0324, "Lake Tahoe, CA"),
    (43.8804, -110.7025, "Jackson Lake, WY"),
    (46.8588, -121.7278, "Mount Rainier, WA"),
    (44.9265, -93.0934, "Minneapolis Lakes, MN"),

    # Forest areas
    (47.6062, -122.3321, "Olympic Peninsula, WA"),
    (35.6762, -105.9378, "Santa Fe National Forest, NM"),
    (34.1479, -118.1445, "Angeles National Forest, CA"),

    # Agricultural/Rural (contrast for classifier)
    (41.8819, -103.6672, "Nebraska Farmland"),
    (38.5816, -121.4944, "Central Valley, CA"),
    (42.4406, -89.0360, "Wisconsin Farmland"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite tiles for scenic locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - downloads to data/tiles/
    python scripts/download_sample_tiles.py

    # Custom output directory
    python scripts/download_sample_tiles.py --output ./my_tiles

    # Higher resolution (zoom 16)
    python scripts/download_sample_tiles.py --zoom 16

    # Download 3x3 grid around each location
    python scripts/download_sample_tiles.py --surrounding 1

    # Limit number of locations (for testing)
    python scripts/download_sample_tiles.py --limit 5

Environment Variables:
    MAPBOX_ACCESS_TOKEN - Your Mapbox API token (required)
                          Get one at: https://account.mapbox.com/access-tokens/
"""
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tiles",
        help="Output directory for tiles (default: data/tiles/)"
    )
    parser.add_argument(
        "--zoom", "-z",
        type=int,
        default=15,
        choices=range(10, 19),
        metavar="[10-18]",
        help="Zoom level (default: 15, higher = more detail)"
    )
    parser.add_argument(
        "--surrounding", "-s",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Download surrounding tiles (0=center only, 1=3x3, 2=5x5)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of locations to download (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Select locations
    locations = SCENIC_LOCATIONS[:args.limit] if args.limit else SCENIC_LOCATIONS

    # Calculate total tiles
    tiles_per_location = (1 + 2 * args.surrounding) ** 2 if args.surrounding else 1
    total_tiles = len(locations) * tiles_per_location

    print("\n" + "=" * 60)
    print("Scenic Drive - Mapbox Tile Downloader")
    print("=" * 60)
    print(f"\nLocations to download: {len(locations)}")
    print(f"Tiles per location: {tiles_per_location} (surrounding={args.surrounding})")
    print(f"Total tiles: {total_tiles}")
    print(f"Zoom level: {args.zoom}")
    print(f"Output directory: {args.output}")

    if args.dry_run:
        print("\n[DRY RUN] Would download tiles for:")
        for lat, lon, name in locations:
            x, y = lat_lon_to_tile(lat, lon, args.zoom)
            print(f"  - {name}: tile ({x}, {y}) at zoom {args.zoom}")
        print("\nRun without --dry-run to actually download.")
        return 0

    # Create tile source
    try:
        source = MapboxTileSource(
            cache_dir=args.output,
            rate_limit=5  # Conservative rate for bulk downloads
        )
    except Exception as e:
        logger.error(f"Failed to initialize MapboxTileSource: {e}")
        return 1

    # Check for token before attempting downloads
    if not source.has_token:
        print("\n" + "!" * 60)
        print("ERROR: No Mapbox access token found!")
        print("!" * 60)
        print("""
To download tiles, you need a Mapbox access token:

1. Go to: https://account.mapbox.com/access-tokens/
2. Create a free account or log in
3. Create a new access token (or use the default public token)
4. Set the environment variable:

   export MAPBOX_ACCESS_TOKEN="pk.your_token_here"

   Or add it to your .env file:
   MAPBOX_ACCESS_TOKEN=pk.your_token_here

Free tier includes 200,000 tile requests per month.
""")
        return 1

    # Validate token
    print("\nValidating Mapbox token...")
    try:
        source.validate_token()
        print("Token validated successfully!")
    except MapboxTokenError as e:
        print(f"\nERROR: Invalid token - {e}")
        return 1
    except MapboxDownloadError as e:
        print(f"\nWARNING: Could not validate token - {e}")
        print("Proceeding anyway...")

    # Download tiles
    print(f"\nDownloading {total_tiles} tiles...")
    print("-" * 60)

    try:
        downloaded_paths = source.download_training_sample(
            locations=locations,
            zoom=args.zoom,
            output_dir=args.output,
            surrounding=args.surrounding
        )
    except MapboxTokenError as e:
        logger.error(f"Token error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        stats = source.get_stats()
        print(f"Partial download: {stats.tiles_downloaded} tiles downloaded")
        return 1

    # Print summary
    stats = source.get_stats()
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Tiles downloaded: {stats.tiles_downloaded}")
    print(f"Tiles from cache: {stats.tiles_cached}")
    print(f"Tiles failed: {stats.tiles_failed}")
    print(f"Cache hit rate: {stats.cache_hit_rate:.1%}")
    print(f"Total time: {stats.total_time_seconds:.1f} seconds")
    if stats.bytes_downloaded > 0:
        mb_downloaded = stats.bytes_downloaded / (1024 * 1024)
        print(f"Data downloaded: {mb_downloaded:.2f} MB")

    print(f"\nTiles saved to: {args.output}")
    print(f"Total files: {len(downloaded_paths)}")

    # List some sample files
    if downloaded_paths:
        print("\nSample files:")
        for path in downloaded_paths[:5]:
            print(f"  - {path.name}")
        if len(downloaded_paths) > 5:
            print(f"  ... and {len(downloaded_paths) - 5} more")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
