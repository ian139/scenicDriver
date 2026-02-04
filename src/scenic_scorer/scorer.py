"""
Scenic Score Calculator
Owner: progno-ml-vision agent

Combines terrain classification with geographic features to produce
a 0-10 scenic beauty score.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TerrainFeatures:
    """Features extracted from terrain analysis (from geospatial agent)."""
    slope_variation: float  # 0-1, terrain roughness
    elevation_change: float  # meters, total elevation range
    water_proximity: float  # 0-1, distance to water (1 = adjacent)
    vegetation_density: float  # 0-1, NDVI-based greenness
    coastal: bool = False
    has_lake: bool = False
    has_river: bool = False


@dataclass
class ClassificationResult:
    """Result from landscape classifier."""
    terrain_class: str
    confidence: float
    scenic_weight: float


class ScenicScorer:
    """
    Calculate scenic beauty scores using a weighted formula.

    Score = base_scenic_weight * confidence
          + terrain_bonus
          + water_bonus
          + vegetation_bonus

    All scores normalized to 0-10 scale.
    """

    # Weights for different components
    CLASSIFICATION_WEIGHT = 0.4
    TERRAIN_WEIGHT = 0.25
    WATER_WEIGHT = 0.2
    VEGETATION_WEIGHT = 0.15

    # Bonus multipliers
    COASTAL_BONUS = 1.5
    LAKE_BONUS = 1.3
    RIVER_BONUS = 1.2
    MOUNTAIN_TERRAIN_BONUS = 1.4

    def calculate_score(
        self,
        classification: ClassificationResult,
        terrain: TerrainFeatures
    ) -> float:
        """
        Calculate the scenic score for a location.

        Args:
            classification: Result from landscape classifier
            terrain: Features from terrain analysis

        Returns:
            Scenic score from 0-10
        """
        # Base score from classification
        base_score = classification.scenic_weight * classification.confidence * 10

        # Terrain variation score (mountainous terrain is more scenic)
        terrain_score = self._calculate_terrain_score(terrain)

        # Water proximity bonus
        water_score = self._calculate_water_score(terrain)

        # Vegetation density score
        vegetation_score = terrain.vegetation_density * 10

        # Weighted combination
        final_score = (
            self.CLASSIFICATION_WEIGHT * base_score +
            self.TERRAIN_WEIGHT * terrain_score +
            self.WATER_WEIGHT * water_score +
            self.VEGETATION_WEIGHT * vegetation_score
        )

        # Apply bonuses for special features
        if terrain.coastal:
            final_score *= self.COASTAL_BONUS
        elif terrain.has_lake:
            final_score *= self.LAKE_BONUS
        elif terrain.has_river:
            final_score *= self.RIVER_BONUS

        if classification.terrain_class == "mountain" and terrain.slope_variation > 0.5:
            final_score *= self.MOUNTAIN_TERRAIN_BONUS

        # Clamp to 0-10
        return float(np.clip(final_score, 0, 10))

    def _calculate_terrain_score(self, terrain: TerrainFeatures) -> float:
        """Score based on terrain variation."""
        # Higher slope variation and elevation change = more scenic
        slope_score = terrain.slope_variation * 6
        elevation_score = min(terrain.elevation_change / 500, 1) * 4  # Cap at 500m
        return slope_score + elevation_score

    def _calculate_water_score(self, terrain: TerrainFeatures) -> float:
        """Score based on water features."""
        base_water = terrain.water_proximity * 8

        # Bonus for specific water features
        if terrain.coastal:
            return 10.0
        elif terrain.has_lake:
            return max(base_water, 8.0)
        elif terrain.has_river:
            return max(base_water, 6.0)

        return base_water


def calculate_scenic_score(
    terrain_class: str,
    confidence: float,
    scenic_weight: float,
    slope_variation: float,
    elevation_change: float,
    water_proximity: float,
    vegetation_density: float = 0.5,
    coastal: bool = False,
    has_lake: bool = False,
    has_river: bool = False
) -> float:
    """
    Convenience function for calculating scenic scores.

    Example:
        score = calculate_scenic_score(
            terrain_class="mountain",
            confidence=0.95,
            scenic_weight=0.95,
            slope_variation=0.8,
            elevation_change=300,
            water_proximity=0.2
        )
    """
    scorer = ScenicScorer()

    classification = ClassificationResult(
        terrain_class=terrain_class,
        confidence=confidence,
        scenic_weight=scenic_weight
    )

    terrain = TerrainFeatures(
        slope_variation=slope_variation,
        elevation_change=elevation_change,
        water_proximity=water_proximity,
        vegetation_density=vegetation_density,
        coastal=coastal,
        has_lake=has_lake,
        has_river=has_river
    )

    return scorer.calculate_score(classification, terrain)
