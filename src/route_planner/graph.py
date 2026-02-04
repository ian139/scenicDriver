"""
Road Network Graph
Owner: progno-backend agent

Graph structure for road network with scenic scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class Node:
    """A node in the road graph (intersection or endpoint)."""
    id: str
    lat: float
    lon: float

    @property
    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class Edge:
    """An edge in the road graph (road segment)."""
    id: str
    start_node_id: str
    end_node_id: str
    distance_km: float
    scenic_score: float  # 0-10
    road_name: Optional[str] = None
    road_type: str = "secondary"  # highway, primary, secondary, residential, scenic_byway
    speed_limit_kmh: int = 50

    @property
    def travel_time_minutes(self) -> float:
        """Estimated travel time at speed limit."""
        return (self.distance_km / self.speed_limit_kmh) * 60


class RoadGraph:
    """
    Road network graph with scenic scores on edges.

    Supports:
    - Spatial queries (find nearest node)
    - Edge traversal
    - Graph I/O (load/save)
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> list of edge_ids

        # Spatial index for nearest-node queries
        self._spatial_index = None

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
        self._spatial_index = None  # Invalidate index

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge

        # Add to adjacency list (bidirectional)
        if edge.start_node_id not in self.adjacency:
            self.adjacency[edge.start_node_id] = []
        self.adjacency[edge.start_node_id].append(edge.id)

        # For undirected graph, add reverse direction
        if edge.end_node_id not in self.adjacency:
            self.adjacency[edge.end_node_id] = []
        # Create reverse edge reference
        reverse_edge_id = f"{edge.id}_rev"
        self.adjacency[edge.end_node_id].append(reverse_edge_id)

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID."""
        return self.nodes[node_id]

    def get_edges(self, node_id: str) -> List[Edge]:
        """Get all edges from a node."""
        edge_ids = self.adjacency.get(node_id, [])
        edges = []

        for edge_id in edge_ids:
            if edge_id.endswith("_rev"):
                # Reverse edge - swap start/end
                original_id = edge_id[:-4]
                original = self.edges[original_id]
                edges.append(Edge(
                    id=edge_id,
                    start_node_id=original.end_node_id,
                    end_node_id=original.start_node_id,
                    distance_km=original.distance_km,
                    scenic_score=original.scenic_score,
                    road_name=original.road_name,
                    road_type=original.road_type,
                    speed_limit_kmh=original.speed_limit_kmh
                ))
            else:
                edges.append(self.edges[edge_id])

        return edges

    def find_nearest_node(self, lat: float, lon: float) -> Node:
        """Find the nearest node to given coordinates."""
        if self._spatial_index is None:
            self._build_spatial_index()

        # TODO: Use spatial index for efficient query
        # For now, brute force
        min_dist = float('inf')
        nearest = None

        for node in self.nodes.values():
            dist = (node.lat - lat)**2 + (node.lon - lon)**2
            if dist < min_dist:
                min_dist = dist
                nearest = node

        if nearest is None:
            raise ValueError("No nodes in graph")

        return nearest

    def _build_spatial_index(self) -> None:
        """Build spatial index for efficient nearest-neighbor queries."""
        # TODO: Implement R-tree or similar spatial index
        # from rtree import index
        # self._spatial_index = index.Index()
        # for node in self.nodes.values():
        #     self._spatial_index.insert(
        #         int(node.id),
        #         (node.lon, node.lat, node.lon, node.lat)
        #     )
        pass

    def save(self, path: Path) -> None:
        """Save graph to JSON file."""
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
                    "speed_limit": e.speed_limit_kmh
                }
                for e in self.edges.values()
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        """Load graph from JSON file."""
        with open(path) as f:
            data = json.load(f)

        graph = cls()

        for node_data in data["nodes"]:
            graph.add_node(Node(
                id=node_data["id"],
                lat=node_data["lat"],
                lon=node_data["lon"]
            ))

        for edge_data in data["edges"]:
            graph.add_edge(Edge(
                id=edge_data["id"],
                start_node_id=edge_data["start"],
                end_node_id=edge_data["end"],
                distance_km=edge_data["distance_km"],
                scenic_score=edge_data["scenic_score"],
                road_name=edge_data.get("road_name"),
                road_type=edge_data.get("road_type", "secondary"),
                speed_limit_kmh=edge_data.get("speed_limit", 50)
            ))

        return graph

    @classmethod
    def from_osm(cls, osm_file: Path, scenic_scores: Dict[str, float] = None) -> "RoadGraph":
        """
        Build graph from OpenStreetMap data.

        Args:
            osm_file: Path to .osm or .pbf file
            scenic_scores: Dict mapping way IDs to scenic scores

        Returns:
            RoadGraph instance
        """
        # TODO: Implement OSM parsing
        # Options:
        # 1. Use osmnx library
        # 2. Parse OSM XML directly
        # 3. Use osmium for .pbf files
        raise NotImplementedError("Implement OSM graph building")
