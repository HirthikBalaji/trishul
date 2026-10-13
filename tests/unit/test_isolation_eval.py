"""Unit tests for Milestone 4: Monte-Carlo connectivity, Restore-first ranking, and Flow-path tracing."""

import networkx as nx
import numpy as np
from rasterio.transform import from_bounds

from trishul.network.montecarlo import MonteCarloConnectivityEvaluator
from trishul.network.restore import RestoreFirstRanker
from trishul.terrain.flowpath import FlowPathTracer


def test_monte_carlo_connectivity():
    """Verify Monte-Carlo connectivity percolation with correlated thresholds."""
    G = nx.Graph()
    # Simple corridor: 0 - 1 - 2
    G.add_edge(0, 1, p_hit=0.9, length=0.01)  # Highly severed edge
    G.add_edge(1, 2, p_hit=0.1, length=0.01)  # Safe edge

    evaluator = MonteCarloConnectivityEvaluator(n_simulations=100)
    settlement_nodes = {"settlement_0": 0, "settlement_1": 1}
    destination_nodes = [2]

    res = evaluator.evaluate_probabilistic_connectivity(G, settlement_nodes, destination_nodes)
    # Settlement 0 is behind edge (0, 1) with p_hit=0.9, so should have high P(cutoff)
    assert res["settlement_0"].p_cutoff > 0.6
    # Settlement 1 only needs edge (1, 2) with p_hit=0.1, so should have low P(cutoff)
    assert res["settlement_1"].p_cutoff < 0.4
    assert 0.0 <= res["settlement_0"].ci_lo <= res["settlement_0"].ci_hi <= 1.0


def test_restore_first_ranking():
    """Verify greedy restore-first ranking prioritizes high-impact bridges."""
    ranker = RestoreFirstRanker()
    severed_edges = [
        {
            "id": "bridge/100",
            "properties": {"name": "Syapru Bridge", "bridge": "yes", "p_hit": 0.85, "length_km": 0.05},
        },
        {
            "id": "way/200",
            "properties": {"name": "Side Spur Road", "highway": "residential", "p_hit": 0.50, "length_km": 0.80},
        },
    ]

    ranked = ranker.rank_repairs(severed_edges, settlements_count=3, buildings_count=23)
    assert len(ranked) == 2
    # Bridge has high buildings reconnected and small repair length -> should be rank 1
    assert ranked[0].rank == 1
    assert ranked[0].asset_type == "bridge"
    assert ranked[0].priority_score > ranked[1].priority_score


def test_flowpath_tracer():
    """Verify downstream flood-path tracing and corridor buffering."""
    tracer = FlowPathTracer(resolution_m=30.0)
    H, W = 80, 80
    elevation = np.zeros((H, W), dtype=np.float32)
    y, x = np.mgrid[0:H, 0:W]
    elevation = 2500.0 - 10.0 * y + 20.0 * np.abs(x - 40)

    stream_mask = np.zeros((H, W), dtype=bool)
    stream_mask[:, 40] = True

    hand_m = np.zeros((H, W), dtype=np.float32)
    for c in range(W):
        hand_m[:, c] = abs(c - 40) * 1.5

    hazard_prob = np.zeros((H, W), dtype=np.float32)
    hazard_prob[:, 38:43] = 0.8

    transform = from_bounds(85.25, 28.10, 85.35, 28.20, W, H)
    osm_places = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [85.30, 28.15]},
                "properties": {"name": "Test Village"},
            }
        ]
    }

    res = tracer.trace_path(
        seed_lon=85.30,
        seed_lat=28.19,
        elevation=elevation,
        stream_mask=stream_mask,
        hand_m=hand_m,
        hazard_prob=hazard_prob,
        transform=transform,
        osm_places=osm_places,
    )

    assert res.path_length_km > 0.0
    assert res.total_drop_m > 0.0
    assert "geometry" in res.geojson_path
    assert res.consistency_overlap_pct >= 0.0
