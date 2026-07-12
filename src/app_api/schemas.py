from __future__ import annotations

from pydantic import BaseModel, Field


class LatLon(BaseModel):
    lat: float
    lon: float


class RouteCompareRequest(BaseModel):
    start: LatLon
    end: LatLon
    scenic_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    region: str = Field(default="pittsfield")
    run_name: str | None = None
    max_detour_factor: float = Field(
        default=1.8, ge=1.0, le=3.0, allow_inf_nan=False
    )
    avoid_highways: bool = False
    include_baseline: bool = True


class ContributorSessionStartRequest(BaseModel):
    contributor_id: str
    display_name: str | None = None
    region: str = "pittsfield"


class ContributorTask(BaseModel):
    task_id: str
    image_path: str
    class_id: int | None = None
    class_name: str | None = None
    scenic_score: float | None = None


class ContributorLabelRequest(BaseModel):
    contributor_id: str
    task_id: str
    image_path: str
    scenic_human: float = Field(ge=0.0, le=10.0)
    confidence: str = "medium"
    skip: bool = False
    notes: str = ""
    region: str = "pittsfield"

