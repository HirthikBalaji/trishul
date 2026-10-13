"""Pre-event and post-event road network graph builder.

In compliance with Competition Rules §6.8:
Constructs a topological network graph from drivable OSM highways and bridges.
Splits edges at intersections, computes traversal weights, and maps asset states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
from shapely.geometry import Point, shape

logger = logging.getLogger(__name__)


@dataclass
class RoadGraphBundle:
    graph: nx.Graph
    node_coords: dict[int, tuple[float, float]]
    edge_features: list[dict[str, Any]]


class RoadGraphBuilder:
    """Builds an undirected topological network graph from OSM road line strings."""

    def __init__(self, tolerance_deg: float = 0.0001):
        self.tolerance_deg = tolerance_deg

    def build_graph(
        self,
        evaluated_roads: list[dict[str, Any]],
        evaluated_bridges: list[dict[str, Any]] | None = None,
    ) -> RoadGraphBundle:
        """Construct NetworkX graph from evaluated roads and bridges."""
        G = nx.Graph()
        node_id_counter = 0
        coord_to_node: dict[tuple[float, float], int] = {}
        node_coords: dict[int, tuple[float, float]] = {}

        def get_or_create_node(pt: tuple[float, float]) -> int:
            nonlocal node_id_counter
            # Quantize coordinate to tolerance
            rounded = (round(pt[0], 5), round(pt[1], 5))
            if rounded not in coord_to_node:
                coord_to_node[rounded] = node_id_counter
                node_coords[node_id_counter] = (pt[0], pt[1])
                G.add_node(node_id_counter, pos=pt)
                node_id_counter += 1
            return coord_to_node[rounded]

        all_edges = list(evaluated_roads) + (evaluated_bridges or [])

        for feat in all_edges:
            geom = shape(feat["geometry"])
            if not hasattr(geom, "coords"):
                continue

            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            props = feat.get("properties", {})
            p_hit = props.get("p_hit", 0.0)
            is_impassable = p_hit >= 0.5

            for i in range(len(coords) - 1):
                u = get_or_create_node(coords[i])
                v = get_or_create_node(coords[i + 1])
                p1, p2 = coords[i], coords[i + 1]
                # Length in degrees
                seg_len = Point(p1).distance(Point(p2))

                G.add_edge(
                    u,
                    v,
                    length=seg_len,
                    p_hit=p_hit,
                    is_impassable=is_impassable,
                    osm_id=props.get("@osmId", "unknown"),
                    name=props.get("name", "Unnamed Road"),
                )

        return RoadGraphBundle(
            graph=G,
            node_coords=node_coords,
            edge_features=all_edges,
        )
