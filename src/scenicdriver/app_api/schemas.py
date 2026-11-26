from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field, model_validator


class LatLon(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    lon: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)


class RouteCompareRequest(BaseModel):
    start: LatLon
    end: LatLon
    region: str = Field(
        default="new_england_north",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    run_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    max_detour_factor: float = Field(
        default=1.8, ge=1.0, le=3.0, allow_inf_nan=False
    )
    avoid_highways: bool = False
    include_baseline: bool = True


class ContributorProfileRequest(BaseModel):
    """Create (POST) or rename (PATCH) a contributor profile."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=128, pattern=r"\S")


class ContributorSessionStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = Field(default="pittsfield", min_length=1, max_length=128)


class ContributorLabelRequest(BaseModel):
    """Identity comes from X-Contributor-Token, never from the body."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    image_path: str = Field(min_length=1, max_length=2048)
    scenic_human: float | None = Field(default=None, ge=0.0, le=10.0, allow_inf_nan=False)
    confidence: str = Field(default="medium", max_length=32)
    skip: bool = False
    notes: str = Field(default="", max_length=4096)
    region: str = Field(default="pittsfield", min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_score_unless_skipping(self) -> "ContributorLabelRequest":
        if not self.skip and self.scenic_human is None:
            raise ValueError("scenic_human is required unless skip is true")
        return self

