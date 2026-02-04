"""
Tests for route planning module.
"""

import pytest
from src.route_planner.graph import RoadGraph, Node, Edge
from src.route_planner.cost import ScenicCostFunction, CostWeights
from src.route_planner.planner import ScenicRoutePlanner


class TestRoadGraph:
    """Test the RoadGraph class."""

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = RoadGraph()
        node = Node(id="1", lat=37.7749, lon=-122.4194)

        graph.add_node(node)

        assert "1" in graph.nodes
        assert graph.get_node("1") == node

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = RoadGraph()
        graph.add_node(Node(id="1", lat=37.7749, lon=-122.4194))
        graph.add_node(Node(id="2", lat=37.7849, lon=-122.4094))

        edge = Edge(
            id="e1",
            start_node_id="1",
            end_node_id="2",
            distance_km=1.5,
            scenic_score=7.5,
            road_type="secondary"
        )
        graph.add_edge(edge)

        assert "e1" in graph.edges

        # Check adjacency
        edges_from_1 = graph.get_edges("1")
        assert len(edges_from_1) == 1
        assert edges_from_1[0].end_node_id == "2"

    def test_bidirectional_edges(self):
        """Test that edges work bidirectionally."""
        graph = RoadGraph()
        graph.add_node(Node(id="1", lat=37.7749, lon=-122.4194))
        graph.add_node(Node(id="2", lat=37.7849, lon=-122.4094))

        edge = Edge(
            id="e1",
            start_node_id="1",
            end_node_id="2",
            distance_km=1.5,
            scenic_score=7.5
        )
        graph.add_edge(edge)

        # Should be traversable both ways
        edges_from_1 = graph.get_edges("1")
        edges_from_2 = graph.get_edges("2")

        assert len(edges_from_1) == 1
        assert len(edges_from_2) == 1

        # Edge from 2 should go to 1
        assert edges_from_2[0].end_node_id == "1"

    def test_find_nearest_node(self):
        """Test finding nearest node to coordinates."""
        graph = RoadGraph()
        graph.add_node(Node(id="1", lat=37.0, lon=-122.0))
        graph.add_node(Node(id="2", lat=38.0, lon=-121.0))
        graph.add_node(Node(id="3", lat=36.0, lon=-123.0))

        # Find node nearest to (37.1, -122.1)
        nearest = graph.find_nearest_node(37.1, -122.1)

        assert nearest.id == "1"


class TestScenicCostFunction:
    """Test the cost function."""

    def test_distance_only(self):
        """With scenic_weight=0, should optimize for distance."""
        cost_fn = ScenicCostFunction(scenic_weight=0.0)

        edge_short = Edge(id="1", start_node_id="a", end_node_id="b",
                         distance_km=1.0, scenic_score=2.0)
        edge_long = Edge(id="2", start_node_id="a", end_node_id="c",
                        distance_km=5.0, scenic_score=9.0)

        # Short edge should have lower cost despite lower scenic score
        assert cost_fn.calculate(edge_short) < cost_fn.calculate(edge_long)

    def test_scenic_only(self):
        """With scenic_weight=1, should optimize for scenery."""
        cost_fn = ScenicCostFunction(scenic_weight=1.0)

        edge_ugly = Edge(id="1", start_node_id="a", end_node_id="b",
                        distance_km=1.0, scenic_score=2.0)
        edge_scenic = Edge(id="2", start_node_id="a", end_node_id="c",
                          distance_km=5.0, scenic_score=9.0)

        # Scenic edge should have lower cost despite being longer
        assert cost_fn.calculate(edge_scenic) < cost_fn.calculate(edge_ugly)

    def test_highway_penalty(self):
        """Test that highways get penalized when avoid_highways=True."""
        cost_fn = ScenicCostFunction(scenic_weight=0.5, avoid_highways=True)

        edge_highway = Edge(id="1", start_node_id="a", end_node_id="b",
                           distance_km=1.0, scenic_score=5.0, road_type="highway")
        edge_secondary = Edge(id="2", start_node_id="a", end_node_id="c",
                             distance_km=1.0, scenic_score=5.0, road_type="secondary")

        # Highway should cost more when avoiding
        assert cost_fn.calculate(edge_highway) > cost_fn.calculate(edge_secondary)


class TestScenicRoutePlanner:
    """Test the route planner."""

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # San Francisco to Los Angeles: ~559 km
        dist = ScenicRoutePlanner._haversine(
            37.7749, -122.4194,  # SF
            34.0522, -118.2437   # LA
        )

        # Should be approximately 559 km
        assert 550 < dist < 570

    def test_simple_route(self):
        """Test route planning on a simple graph."""
        # Create a simple graph: A -> B -> C
        graph = RoadGraph()
        graph.add_node(Node(id="A", lat=37.0, lon=-122.0))
        graph.add_node(Node(id="B", lat=37.5, lon=-122.0))
        graph.add_node(Node(id="C", lat=38.0, lon=-122.0))

        graph.add_edge(Edge(
            id="AB", start_node_id="A", end_node_id="B",
            distance_km=55, scenic_score=6.0
        ))
        graph.add_edge(Edge(
            id="BC", start_node_id="B", end_node_id="C",
            distance_km=55, scenic_score=8.0
        ))

        planner = ScenicRoutePlanner(graph=graph)
        route = planner.find_scenic_route(
            start=(37.0, -122.0),
            end=(38.0, -122.0),
            scenic_weight=0.5
        )

        assert len(route.segments) == 2
        assert route.total_distance_km == 110
