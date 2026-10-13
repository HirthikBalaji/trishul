"""Settlement isolation and connectivity analysis module.

In compliance with Competition Rules §6.8:
Evaluates pre-event road connectivity and identifies settlements severed from hospitals
and major hubs following edge failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
from shapely.geometry import Point, shape

from trishul.network.graph import RoadGraphBundle

logger = logging.getLogger(__name__)


@dataclass
class SettlementIsolationResult:
    settlement_id: str
    name: str
    name_ne: str | None
    status: str  # "reachable", "cut off by road", "no road access (pre-existing)"
    p_cutoff: float
    nearest_destination: str | None
    coordinates: list[float]  # [lon, lat]


@dataclass
class NetworkAnalysisBundle:
    total_settlements: int
    cut_off_settlements: int
    reachable_settlements: int
    pre_existing_unconnected: int
    settlements: list[SettlementIsolationResult]
    destinations: list[str]


class NetworkConnectivityAnalyzer:
    """Analyzes reachability of mountain settlements to hospitals on post-flood graphs."""

    def __init__(self, snap_radius_deg: float = 0.005):  # ~500m in degrees
        self.snap_radius_deg = snap_radius_deg

    def analyze(
        self,
        graph_bundle: RoadGraphBundle,
        osm_places: dict[str, Any],
        osm_facilities: dict[str, Any],
    ) -> NetworkAnalysisBundle:
        """Run deterministic connectivity analysis."""
        G = graph_bundle.graph
        node_coords = graph_bundle.node_coords

        # 1. Identify destinations (Hospitals, major towns)
        dest_nodes = []
        dest_names = []
        for feat in osm_facilities.get("features", []):
            geom = shape(feat["geometry"])
            pt = geom.centroid if geom.geom_type != "Point" else geom
            props = feat.get("properties", {})
            name = props.get("name") or props.get("amenity") or "Hospital"
            # Snap to graph
            nearest_n = self._find_nearest_node(pt.x, pt.y, node_coords)
            if nearest_n is not None:
                dest_nodes.append(nearest_n)
                dest_names.append(name)

        if not dest_nodes:
            # Fallback to extreme node of the graph (e.g. lowest elevation / downstream town)
            if len(G.nodes) > 0:
                downstream_node = list(G.nodes)[0]
                dest_nodes.append(downstream_node)
                dest_names.append("Downstream Hub")

        # 2. Build post-event graph by filtering out severed edges (p_hit >= 0.5)
        G_post = nx.Graph()
        G_post.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            if not data.get("is_impassable", False):
                G_post.add_edge(u, v, **data)

        # 3. Evaluate settlements
        place_features = osm_places.get("features", [])
        settlement_results: list[SettlementIsolationResult] = []
        cut_off_count = 0
        reachable_count = 0
        pre_existing_count = 0

        for place in place_features:
            geom = shape(place["geometry"])
            pt = geom.centroid if geom.geom_type != "Point" else geom
            props = place.get("properties", {})
            name = props.get("name", "Unnamed Settlement")
            name_ne = props.get("name:ne")
            settlement_id = place.get("id", f"place/{len(settlement_results)}")

            # Check if connected in PRE-event graph
            nearest_node = self._find_nearest_node(pt.x, pt.y, node_coords)
            if nearest_node is None:
                # Never had pre-event road connection! Rule §6.8 explicitly forbids calling this cut-off
                status = "no road access (pre-existing)"
                p_cutoff = 0.0
                pre_existing_count += 1
            else:
                # Check reachability in post-event graph to any destination
                reachable = False
                for d_node in dest_nodes:
                    if nx.has_path(G_post, nearest_node, d_node):
                        reachable = True
                        break

                if reachable:
                    status = "reachable"
                    p_cutoff = 0.05
                    reachable_count += 1
                else:
                    status = "cut off by road"
                    p_cutoff = 0.95
                    cut_off_count += 1

            settlement_results.append(
                SettlementIsolationResult(
                    settlement_id=settlement_id,
                    name=name,
                    name_ne=name_ne,
                    status=status,
                    p_cutoff=p_cutoff,
                    nearest_destination=dest_names[0] if dest_names else None,
                    coordinates=[round(pt.x, 5), round(pt.y, 5)],
                )
            )

        return NetworkAnalysisBundle(
            total_settlements=len(settlement_results),
            cut_off_settlements=cut_off_count,
            reachable_settlements=reachable_count,
            pre_existing_unconnected=pre_existing_count,
            settlements=settlement_results,
            destinations=dest_names,
        )

    def _find_nearest_node(
        self,
        x: float,
        y: float,
        node_coords: dict[int, tuple[float, float]],
    ) -> int | None:
        """Find the closest graph node within tolerance."""
        best_node = None
        best_dist = float("inf")
        target = Point(x, y)

        for n, coords in node_coords.items():
            dist = target.distance(Point(coords))
            if dist < best_dist and dist <= self.snap_radius_deg:
                best_dist = dist
                best_node = n

        return best_node
