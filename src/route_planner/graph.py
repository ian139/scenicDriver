from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Node:
    id: str
    lat: float
    lon: float

    @property
    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class Edge:
    id: str
    start_node_id: str
    end_node_id: str
    distance_km: float
    scenic_score: float
    road_name: Optional[str] = None
    road_type: str = "secondary"
    speed_limit_kmh: int = 50

    @property
    def travel_time_minutes(self) -> float:
        speed = max(float(self.speed_limit_kmh), 1.0)
        return (self.distance_km / speed) * 60.0


class RoadGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        # node_id -> list[(edge_id, is_reverse_view)]
        self.adjacency: Dict[str, List[Tuple[str, bool]]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        if edge.start_node_id not in self.nodes:
            raise ValueError(f"Unknown start node: {edge.start_node_id}")
        if edge.end_node_id not in self.nodes:
            raise ValueError(f"Unknown end node: {edge.end_node_id}")
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge.id}")

        self.edges[edge.id] = edge
        self.adjacency.setdefault(edge.start_node_id, []).append((edge.id, False))
        self.adjacency.setdefault(edge.end_node_id, []).append((edge.id, True))

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def get_edges(self, node_id: str) -> List[Edge]:
        out: List[Edge] = []
        for edge_id, reverse in self.adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            if not reverse:
                out.append(edge)
                continue
            out.append(
                Edge(
                    id=f"{edge.id}::rev",
                    start_node_id=edge.end_node_id,
                    end_node_id=edge.start_node_id,
                    distance_km=edge.distance_km,
                    scenic_score=edge.scenic_score,
                    road_name=edge.road_name,
                    road_type=edge.road_type,
                    speed_limit_kmh=edge.speed_limit_kmh,
                )
            )
        return out

    def find_nearest_node(self, lat: float, lon: float) -> Node:
        if not self.nodes:
            raise ValueError("Road graph has no nodes")
        best: Optional[Node] = None
        best_dist = float("inf")
        for node in self.nodes.values():
            d = (node.lat - lat) ** 2 + (node.lon - lon) ** 2
            if d < best_dist:
                best_dist = d
                best = node
        assert best is not None
        return best

    def save(self, path: Path) -> None:
        data = {
            "nodes": [
                {"id": n.id, "lat": n.lat, "lon": n.lon}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "start": e.start_node_id,
                    "end": e.end_node_id,
                    "distance_km": e.distance_km,
                    "scenic_score": e.scenic_score,
                    "road_name": e.road_name,
                    "road_type": e.road_type,
                    "speed_limit_kmh": e.speed_limit_kmh,
                }
                for e in self.edges.values()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = cls()
        for row in data.get("nodes", []):
            graph.add_node(Node(id=str(row["id"]), lat=float(row["lat"]), lon=float(row["lon"])))
        for row in data.get("edges", []):
            graph.add_edge(
                Edge(
                    id=str(row["id"]),
                    start_node_id=str(row["start"]),
                    end_node_id=str(row["end"]),
                    distance_km=float(row["distance_km"]),
                    scenic_score=float(row.get("scenic_score", 5.0)),
                    road_name=row.get("road_name"),
                    road_type=str(row.get("road_type", "secondary")),
                    speed_limit_kmh=int(row.get("speed_limit_kmh", 50)),
                )
            )
        return graph

    @classmethod
    def from_geojson(cls, path: Path) -> "RoadGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "FeatureCollection":
            raise ValueError("GeoJSON must be a FeatureCollection")

        graph = cls()
        node_index: Dict[Tuple[float, float], str] = {}
        edge_counter = 0

        def node_id(lat: float, lon: float) -> str:
            key = (round(float(lat), 7), round(float(lon), 7))
            if key in node_index:
                return node_index[key]
            nid = f"n{len(node_index)}"
            node_index[key] = nid
            graph.add_node(Node(id=nid, lat=float(lat), lon=float(lon)))
            return nid

        for feat in data.get("features", []):
            geom = feat.get("geometry", {}) or {}
            if geom.get("type") != "LineString":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            props = feat.get("properties", {}) or {}
            base_edge_id = props.get("id")
            road_name = props.get("road_name")
            road_type = str(props.get("road_type", "secondary"))
            scenic_score = float(props.get("scenic_score", 5.0))
            speed_limit = int(props.get("speed_limit_kmh", 50))

            for idx in range(len(coords) - 1):
                lon1, lat1 = coords[idx]
                lon2, lat2 = coords[idx + 1]
                start = node_id(lat1, lon1)
                end = node_id(lat2, lon2)
                dist_km = _haversine_km(lat1, lon1, lat2, lon2)
                edge_id = f"{base_edge_id}_{idx}" if base_edge_id else f"e{edge_counter}"
                edge_counter += 1
                graph.add_edge(
                    Edge(
                        id=edge_id,
                        start_node_id=start,
                        end_node_id=end,
                        distance_km=float(dist_km),
                        scenic_score=float(scenic_score),
                        road_name=road_name,
                        road_type=road_type,
                        speed_limit_kmh=speed_limit,
                    )
                )

        if not graph.nodes or not graph.edges:
            raise ValueError(f"No usable LineString edges found in {path}")
        return graph

    @classmethod
    def from_osm(cls, osm_file: Path, scenic_scores: Optional[Dict[str, float]] = None) -> "RoadGraph":
        try:
            import osmnx as ox
        except ImportError as exc:
            raise ImportError("osmnx is required for OSM import. Run: uv sync --extra geo") from exc

        scenic_scores = scenic_scores or {}
        osm_path = Path(osm_file)
        if osm_path.suffix == ".graphml":
            G = ox.load_graphml(osm_path)
        else:
            G = ox.graph_from_xml(osm_path)
        return _graph_from_osmnx(G, scenic_scores)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def _graph_from_osmnx(G: Any, scenic_scores: Dict[str, float]) -> RoadGraph:
    graph = RoadGraph()

    for node_id, data in G.nodes(data=True):
        graph.add_node(
            Node(
                id=str(node_id),
                lat=float(data.get("y")),
                lon=float(data.get("x")),
            )
        )

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_id = f"{u}-{v}-{key}"
        length_m = float(data.get("length", 0.0))
        scenic_score = float(scenic_scores.get(str(data.get("osmid", edge_id)), 5.0))
        road_name = data.get("name")
        road_type = data.get("highway", "secondary")
        if isinstance(road_type, list):
            road_type = road_type[0]

        speed = data.get("maxspeed", 50)
        if isinstance(speed, list):
            speed = speed[0]
        try:
            speed_kmh = int(str(speed).split()[0])
        except (TypeError, ValueError):
            speed_kmh = 50

        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id=str(u),
                end_node_id=str(v),
                distance_km=max(0.0, length_m / 1000.0),
                scenic_score=float(max(0.0, min(10.0, scenic_score))),
                road_name=road_name,
                road_type=str(road_type),
                speed_limit_kmh=speed_kmh,
            )
        )

    return graph
