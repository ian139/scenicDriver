from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
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
    one_way: bool = False

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
        if not edge.one_way:
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
                    one_way=False,
                )
            )
        return out

    def find_nearest_node(self, lat: float, lon: float) -> Node:
        return self.find_nearest_node_with_distance(lat, lon)[0]

    def find_nearest_node_with_distance(self, lat: float, lon: float) -> tuple[Node, float]:
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
        return best, _haversine_km(float(lat), float(lon), best.lat, best.lon)

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
                    "one_way": e.one_way,
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
                    speed_limit_kmh=_parse_speed_limit_kmh(
                        row.get("speed_limit_kmh"),
                        str(row.get("road_type", "secondary")),
                    ),
                    # Historical graph JSONs often omitted one_way.
                    # Default to True to preserve OSM directed-edge semantics.
                    one_way=_parse_one_way(row.get("one_way"), default=True),
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
            road_type = _normalize_road_type(props.get("road_type", "secondary"))
            scenic_score = float(props.get("scenic_score", 5.0))
            speed_limit = _parse_speed_limit_kmh(props.get("speed_limit_kmh"), road_type)
            one_way = _parse_one_way(props.get("one_way", props.get("oneway")), default=True)
            if "bidirectional" in props:
                one_way = not _parse_bool(props.get("bidirectional"), default=False)

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
                        one_way=one_way,
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
        road_type = _normalize_road_type(data.get("highway", "secondary"))
        speed_kmh = _parse_speed_limit_kmh(data.get("maxspeed"), road_type)
        # osmnx graphs are directed. Keep this edge as directed by default
        # and only synthesize reverse traversal when we are sure it's needed.
        one_way = _parse_one_way(data.get("oneway"), default=True)
        if not one_way and G.has_edge(v, u):
            one_way = True

        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id=str(u),
                end_node_id=str(v),
                distance_km=max(0.0, length_m / 1000.0),
                scenic_score=float(max(0.0, min(10.0, scenic_score))),
                road_name=road_name,
                road_type=road_type,
                speed_limit_kmh=speed_kmh,
                one_way=one_way,
            )
        )

    return graph


_ROAD_TYPE_SPEED_KMH = {
    "motorway": 100,
    "trunk": 90,
    "primary": 80,
    "secondary": 60,
    "tertiary": 50,
    "residential": 35,
    "service": 25,
    "unclassified": 40,
    "living_street": 20,
}


def _normalize_road_type(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        raw = raw[0]
    value = str(raw).strip().lower() if raw is not None else ""
    return value if value else "secondary"


def _parse_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if text in {"0", "false", "no", "n", "f", "off"}:
        return False
    return default


def _parse_one_way(raw: Any, default: bool) -> bool:
    """
    Parse one-way flags from OSM/GeoJSON-style values.

    OSM may encode oneway as yes/no/1/0/-1.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return default

    text = str(raw).strip().lower()
    if text in {"-1", "reverse"}:
        return True
    return _parse_bool(raw, default=default)


def _parse_speed_limit_kmh(raw_speed: Any, road_type: str) -> int:
    """
    Parse OSM maxspeed into km/h.

    Handles values like:
      - 50
      - "50"
      - "35 mph"
      - "50 km/h"
      - "50;70"
      - ["50 mph", ...]
    Falls back to road-type defaults when missing/unparseable.
    """
    fallback = int(_ROAD_TYPE_SPEED_KMH.get(road_type, 50))
    if raw_speed is None:
        return fallback

    if isinstance(raw_speed, list):
        raw_speed = raw_speed[0] if raw_speed else None
        if raw_speed is None:
            return fallback

    if isinstance(raw_speed, (int, float)):
        speed = float(raw_speed)
        if speed <= 0:
            return fallback
        return int(round(min(max(speed, 10.0), 140.0)))

    text = str(raw_speed).strip().lower()
    if not text:
        return fallback
    if ";" in text:
        text = text.split(";", 1)[0].strip()

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return fallback

    value = float(match.group(0))
    if value <= 0:
        return fallback

    if "mph" in text:
        value *= 1.60934

    value = min(max(value, 10.0), 140.0)
    return int(round(value))
