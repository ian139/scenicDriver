"""
API Configuration
Owner: progno-backend agent
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    debug: bool = False
    api_prefix: str = "/api/v1"

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Database
    database_url: str = "postgresql://localhost:5432/progno"

    # Redis (for caching)
    redis_url: str = "redis://localhost:6379"

    # External APIs
    mapbox_token: str = ""

    # Model paths
    classifier_model_path: str = "./models/classifier/best_model.pt"
    regression_model_path: str = "./models/regression/best_model.pt"

    # Performance
    route_timeout_seconds: float = 10.0
    max_route_distance_km: float = 2000.0

    class Config:
        env_file = ".env"
        env_prefix = "PROGNO_"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
