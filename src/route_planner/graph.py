from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    id: str
    lat: float
    lon: float


@dataclass
class Edge:
    id: str
    start_node_id: str
    end_node_id: str
    distance_km: float
    scenic_score: float
