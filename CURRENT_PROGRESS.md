# Current Progress

## S3 tile acquisition

The `new_england_north` z14 Mapbox tile acquisition is complete in S3.

Canonical S3 prefixes:

- `s3://scenicdriver-data/raw/images/satellite/z14/new_england_north/`
- `s3://scenicdriver-data/raw/images/terrain/z14/new_england_north/`

Downloaded object counts:

- Satellite: `98,838` PNG tiles
- Terrain RGB: `98,838` PNG tiles
- Total Mapbox raster requests completed: `197,676`

Tile coordinate coverage:

- Zoom: `14`
- x range: `4846..5151`
- y range: `5729..6051`
- Requested bbox: `min_lat=42.5`, `min_lon=-73.5`, `max_lat=47.5`, `max_lon=-66.8`
- Actual tile coverage: `min_lat=42.488301979602255`, `min_lon=-73.5205078125`, `max_lat=47.50235895196859`, `max_lon=-66.796875`

Corner tiles were verified present for both layers:

- `4846_5729.png`
- `4846_6051.png`
- `5151_5729.png`
- `5151_6051.png`

## Notes

- The S3 prefixes were empty before the completed run.
- Partial uploads from cancelled early attempts were removed before the final download.
- No Mapbox token or AWS credential is stored in this repository.

## Next step

Generate heuristic labels and the report for the region:

```bash
uv run python scripts/reports/heuristic_report_region.py \
  --region new_england_north \
  --zoom 14 \
  --raw-dir s3://scenicdriver-data/raw \
  --s3-only \
  --write-labels
```
