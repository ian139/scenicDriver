from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class Tile:
    """Tile payload used by download/feature pipelines."""

    image: np.ndarray
    lat: float
    lon: float
    zoom: int
    x: int
    y: int
    source: str

    @property
    def tile_id(self) -> str:
        return f"{self.source}_{self.zoom}_{self.x}_{self.y}"

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.image)


