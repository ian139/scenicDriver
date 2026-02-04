"""
Tests for scenic scoring module.
"""

import pytest
from src.scenic_scorer.scorer import (
    ScenicScorer,
    TerrainFeatures,
    ClassificationResult,
    calculate_scenic_score
)


class TestScenicScorer:
    """Test the ScenicScorer class."""

    def test_mountain_high_score(self):
        """Mountains with high confidence should score well."""
        scorer = ScenicScorer()

        classification = ClassificationResult(
            terrain_class="mountain",
            confidence=0.95,
            scenic_weight=0.95
        )

        terrain = TerrainFeatures(
            slope_variation=0.8,
            elevation_change=500,
            water_proximity=0.0,
            vegetation_density=0.6,
            coastal=False,
            has_lake=False,
            has_river=False
        )

        score = scorer.calculate_score(classification, terrain)

        # Mountain terrain should score highly
        assert score >= 7.0
        assert score <= 10.0

    def test_highway_low_score(self):
        """Highway areas should score low."""
        scorer = ScenicScorer()

        classification = ClassificationResult(
            terrain_class="freeway",
            confidence=0.9,
            scenic_weight=0.1
        )

        terrain = TerrainFeatures(
            slope_variation=0.1,
            elevation_change=10,
            water_proximity=0.0,
            vegetation_density=0.2,
            coastal=False,
            has_lake=False,
            has_river=False
        )

        score = scorer.calculate_score(classification, terrain)

        # Highway should score low
        assert score < 4.0

    def test_coastal_bonus(self):
        """Coastal areas should get a bonus."""
        scorer = ScenicScorer()

        classification = ClassificationResult(
            terrain_class="beach",
            confidence=0.9,
            scenic_weight=0.9
        )

        terrain_coastal = TerrainFeatures(
            slope_variation=0.2,
            elevation_change=20,
            water_proximity=1.0,
            vegetation_density=0.3,
            coastal=True,
            has_lake=False,
            has_river=False
        )

        terrain_inland = TerrainFeatures(
            slope_variation=0.2,
            elevation_change=20,
            water_proximity=0.0,
            vegetation_density=0.3,
            coastal=False,
            has_lake=False,
            has_river=False
        )

        score_coastal = scorer.calculate_score(classification, terrain_coastal)
        score_inland = scorer.calculate_score(classification, terrain_inland)

        # Coastal should score higher
        assert score_coastal > score_inland

    def test_score_bounds(self):
        """Scores should always be between 0 and 10."""
        scorer = ScenicScorer()

        # Test with extreme values
        extreme_high = ClassificationResult(
            terrain_class="mountain",
            confidence=1.0,
            scenic_weight=1.0
        )

        extreme_terrain = TerrainFeatures(
            slope_variation=1.0,
            elevation_change=2000,
            water_proximity=1.0,
            vegetation_density=1.0,
            coastal=True,
            has_lake=True,
            has_river=True
        )

        score = scorer.calculate_score(extreme_high, extreme_terrain)
        assert 0 <= score <= 10


class TestConvenienceFunction:
    """Test the calculate_scenic_score convenience function."""

    def test_basic_calculation(self):
        """Test the convenience function works."""
        score = calculate_scenic_score(
            terrain_class="forest",
            confidence=0.85,
            scenic_weight=0.85,
            slope_variation=0.4,
            elevation_change=100,
            water_proximity=0.3
        )

        assert isinstance(score, float)
        assert 0 <= score <= 10

    def test_with_water_features(self):
        """Test scoring with water features."""
        score_with_lake = calculate_scenic_score(
            terrain_class="meadow",
            confidence=0.8,
            scenic_weight=0.8,
            slope_variation=0.3,
            elevation_change=50,
            water_proximity=0.8,
            has_lake=True
        )

        score_without = calculate_scenic_score(
            terrain_class="meadow",
            confidence=0.8,
            scenic_weight=0.8,
            slope_variation=0.3,
            elevation_change=50,
            water_proximity=0.0,
            has_lake=False
        )

        assert score_with_lake > score_without
