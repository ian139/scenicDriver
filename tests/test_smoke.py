from src.data_pipeline.mapbox import lat_lon_to_tile


def test_lat_lon_to_tile_shape() -> None:
    x, y = lat_lon_to_tile(42.36, -72.52, 16)
    assert isinstance(x, int)
    assert isinstance(y, int)
