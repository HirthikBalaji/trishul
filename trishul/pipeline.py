"""TRISHUL Pipeline — End-to-End Orchestrator.

In compliance with Competition Rules §0 Rule 2 & §7:
Single entry point: run(aoi, event_date, output_dir, ...)
Produces the complete run bundle:
- manifest.json (Compliance verified)
- facts.json (Claim Ledger)
- hazard_prob.tif, hazard_class.tif, reliability.tif, blindspot.tif
- assets.geojson / assets.gpkg
- network.json
- report_en.pdf, report_ne.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from trishul.compliance.manifest import ProductRecord, RunManifest
from trishul.exposure.bridges import BridgeExposureEvaluator
from trishul.exposure.buildings import BuildingExposureEvaluator
from trishul.exposure.roads import RoadExposureEvaluator
from trishul.fusion.postprocess import FusionPostprocessor
from trishul.ingest.cdse import CDSEClient
from trishul.ingest.dem import DemManager
from trishul.ingest.ohsome import OhsomeClient
from trishul.network.connectivity import NetworkConnectivityAnalyzer
from trishul.network.graph import RoadGraphBuilder
from trishul.report.ledger import ClaimLedger
from trishul.report.pdf import SituationReportPdfGenerator
from trishul.sar.backends import DefaultSarBackend
from trishul.sar.change import SarChangeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trishul.pipeline")

# Trishuli Valley August 2026 Case Study Default Bounding Box
# Rasuwa District, Bhote Koshi / Trishuli river corridor (Syapru Besi to Betrawati)
TRISHULI_CASE_STUDY_BBOX = [85.25, 28.08, 85.42, 28.24]
TRISHULI_EVENT_DATE = "2026-08-26"


def parse_aoi_bounds(aoi_input: Any) -> list[float]:
    """Parse various AOI representations (bbox list, GeoJSON dict, or GeoJSON string) into [min_x, min_y, max_x, max_y]."""
    if isinstance(aoi_input, str):
        # Could be JSON string or comma-separated numbers
        try:
            parsed = json.loads(aoi_input)
            return parse_aoi_bounds(parsed)
        except json.JSONDecodeError:
            parts = [float(p.strip()) for p in aoi_input.split(",")]
            if len(parts) == 4:
                return parts
            raise ValueError(f"Could not parse bbox from string: {aoi_input}") from None

    if isinstance(aoi_input, (list, tuple)) and len(aoi_input) == 4:
        return [float(x) for x in aoi_input]

    if isinstance(aoi_input, dict):
        geom = aoi_input.get("geometry", aoi_input)
        s = shape(geom)
        return list(s.bounds)

    raise ValueError(f"Unsupported AOI input type: {type(aoi_input)}")


def run_pipeline(
    aoi: Any,
    event_date: str,
    output_dir: Path | str,
    run_id: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute end-to-end TRISHUL processing pipeline."""
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Starting TRISHUL Pipeline Run [{run_id}] ===")
    logger.info(f"Event Date: {event_date} | Output Directory: {out_dir}")

    bounds = parse_aoi_bounds(aoi)
    logger.info(f"AOI Bounding Box: {bounds}")

    # 1. Ingest DEM
    logger.info("[Stage 1/7] Ingesting Copernicus DEM (GLO-30)...")
    dem_mgr = DemManager()
    elevation, transform, crs = dem_mgr.get_dem_for_bounds(bounds)

    # 2. Ingest Pre-Event OSM Snapshot
    logger.info("[Stage 2/7] Fetching pre-event OSM assets via Ohsome API...")
    ohsome = OhsomeClient()
    osm_snapshot_date = ohsome.get_latest_valid_snapshot(event_date)
    logger.info(f"Verified pre-event OSM snapshot date: {osm_snapshot_date}")

    roads_geojson = ohsome.fetch_elements(bounds, "highway=* and geometry:line", osm_snapshot_date, "roads")
    bridges_geojson = ohsome.fetch_elements(bounds, "(bridge=* or man_made=bridge)", osm_snapshot_date, "bridges")
    buildings_geojson = ohsome.fetch_elements(bounds, "building=* and geometry:polygon", osm_snapshot_date, "buildings")
    places_geojson = ohsome.fetch_elements(bounds, "place in (city,town,village,hamlet)", osm_snapshot_date, "places")
    facilities_geojson = ohsome.fetch_elements(bounds, "amenity=hospital or healthcare=hospital", osm_snapshot_date, "facilities")

    # 3. Sentinel-1 SAR Same-Track Pair Ingest & Change Detection
    logger.info("[Stage 3/7] Ingesting same-track Sentinel-1 SAR and computing change statistics...")
    cdse = CDSEClient(allow_mock=True)
    sar_backend = DefaultSarBackend(cdse_client=cdse)
    pre_s1, post_s1, s1_stack = sar_backend.get_same_track_pair(bounds, event_date, elevation, transform, crs)

    detector = SarChangeDetector()
    change_result = detector.detect_change(pre_s1, post_s1, s1_stack)

    # 4. Evidence Fusion & Raster Export
    logger.info("[Stage 4/7] Fusing multi-evidence posteriors and exporting GeoTIFFs...")
    postprocessor = FusionPostprocessor()
    fusion_outputs = postprocessor.process(
        water_llr=change_result.water_llr,
        debris_llr=change_result.debris_llr,
        reliability=change_result.reliability,
    )
    raster_paths = postprocessor.export_rasters(fusion_outputs, transform, crs, out_dir)

    # 5. Infrastructure Exposure Analysis (Buildings, Roads, Bridges)
    logger.info("[Stage 5/7] Evaluating infrastructure exposure (Prong 2)...")
    bld_evaluator = BuildingExposureEvaluator()
    bld_eval = bld_evaluator.evaluate(buildings_geojson, fusion_outputs.hazard_prob, transform)

    road_evaluator = RoadExposureEvaluator()
    road_eval = road_evaluator.evaluate(roads_geojson, fusion_outputs.hazard_prob, transform)

    bridge_evaluator = BridgeExposureEvaluator()
    bridge_eval = bridge_evaluator.evaluate(bridges_geojson, fusion_outputs.hazard_prob, transform)

    # Combine assets into unified GeoJSON
    all_asset_features = (
        bld_eval.features + road_eval.features + bridge_eval.features
    )
    assets_bundle = {
        "type": "FeatureCollection",
        "features": all_asset_features,
    }
    assets_path = out_dir / "assets.geojson"
    with open(assets_path, "w", encoding="utf-8") as f:
        json.dump(assets_bundle, f)

    # 6. Topological Network & Settlement Isolation Analysis (Prong 3)
    logger.info("[Stage 6/7] Building topological road graph and analyzing isolation (Prong 3)...")
    graph_builder = RoadGraphBuilder()
    graph_bundle = graph_builder.build_graph(road_eval.features, bridge_eval.features)

    connectivity_analyzer = NetworkConnectivityAnalyzer()
    net_bundle = connectivity_analyzer.analyze(graph_bundle, places_geojson, facilities_geojson)

    network_path = out_dir / "network.json"
    with open(network_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_settlements": net_bundle.total_settlements,
            "cut_off_settlements": net_bundle.cut_off_settlements,
            "reachable_settlements": net_bundle.reachable_settlements,
            "pre_existing_unconnected": net_bundle.pre_existing_unconnected,
            "settlements": [s.__dict__ for s in net_bundle.settlements],
            "destinations": net_bundle.destinations,
        }, f, indent=2)

    # 7. Auditable Claim Ledger and Situation Report Generation
    logger.info("[Stage 7/7] Registering facts in Claim Ledger and generating situation reports...")
    ledger = ClaimLedger(run_id=run_id)
    ledger.register("flood_water_km2", fusion_outputs.flood_water_km2, "km²", layer="hazard", method="sar_change")
    ledger.register("debris_deposit_km2", fusion_outputs.debris_deposit_km2, "km²", layer="hazard", method="sar_change")
    ledger.register("total_affected_km2", fusion_outputs.total_affected_km2, "km²", layer="hazard", method="fusion")
    ledger.register("blindspot_pct", fusion_outputs.blindspot_pct, "%", layer="hazard", method="geometry_gate")
    ledger.register("buildings_total", bld_eval.total_buildings, "structures", layer="exposure", method="osm_overlay")
    ledger.register("buildings_hit_expected", bld_eval.expected_hit_count, "structures", layer="exposure", method="prob_hit")
    ledger.register("road_length_total_km", road_eval.total_length_km, "km", layer="exposure", method="network")
    ledger.register("road_length_hit_km", road_eval.total_length_hit_km, "km", layer="exposure", method="network")
    ledger.register("bridges_total", bridge_eval.total_bridges, "bridges", layer="exposure", method="osm_overlay")
    ledger.register("bridges_hit_count", bridge_eval.confirmed_hit_count, "bridges", layer="exposure", method="noisy_or")
    ledger.register("settlements_total", net_bundle.total_settlements, "settlements", layer="network", method="clustering")
    ledger.register("settlements_cutoff_count", net_bundle.cut_off_settlements, "settlements", layer="network", method="percolation")
    ledger.register("settlements_reachable_count", net_bundle.reachable_settlements, "settlements", layer="network", method="percolation")

    facts_path = ledger.export_json(out_dir)

    # PDF Reports
    pdf_gen = SituationReportPdfGenerator(ledger)
    pdf_en = pdf_gen.generate_pdf(out_dir / "report_en.pdf", lang="en", event_date=event_date)
    pdf_ne = pdf_gen.generate_pdf(out_dir / "report_ne.pdf", lang="ne", event_date=event_date)

    # Compliance Manifest
    manifest = RunManifest(
        run_id=run_id,
        event_date=event_date,
        osm_snapshot_date=osm_snapshot_date,
        aoi_bounds=bounds,
        products=[
            ProductRecord(
                sensor="Sentinel-1",
                product_id=post_s1.scene_id,
                acquisition_time=post_s1.acquisition_time,
                relative_orbit=post_s1.relative_orbit,
                orbit_direction=post_s1.orbit_direction,
            ),
            ProductRecord(
                sensor="Copernicus-DEM",
                product_id="Copernicus_GLO_30",
                acquisition_time="2026-01-01",
            ),
        ],
    )
    manifest_path = manifest.save(out_dir)

    logger.info(f"=== Pipeline Run [{run_id}] Finished Successfully ===")
    logger.info(f"Run Manifest Verdict: {manifest.compliance_verdict}")
    logger.info(f"Deliverables located in: {out_dir}")

    return {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "facts": str(facts_path),
        "assets": str(assets_path),
        "network": str(network_path),
        "report_en": str(pdf_en),
        "report_ne": str(pdf_ne),
        "rasters": {k: str(v) for k, v in raster_paths.items()},
        "metrics": {
            "total_affected_km2": fusion_outputs.total_affected_km2,
            "buildings_hit_expected": bld_eval.expected_hit_count,
            "road_length_hit_km": road_eval.total_length_hit_km,
            "cut_off_settlements": net_bundle.cut_off_settlements,
        },
    }


def main() -> None:
    """CLI entry point for running TRISHUL pipeline."""
    parser = argparse.ArgumentParser(description="TRISHUL: Multimodal Flood Damage Mapping from Space")
    parser.add_argument("--aoi", type=str, help="AOI as bbox 'min_x,min_y,max_x,max_y' or GeoJSON string")
    parser.add_argument("--event-date", type=str, default=TRISHULI_EVENT_DATE, help="Event ISO date (YYYY-MM-DD)")
    parser.add_argument("--case-study", action="store_true", help="Run Trishuli Valley August 2026 case study")
    parser.add_argument("--output", type=str, default="./runs/latest", help="Output directory")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ne"], help="Report language")

    args = parser.parse_args()

    if args.case_study:
        aoi = TRISHULI_CASE_STUDY_BBOX
        event_date = TRISHULI_EVENT_DATE
    elif args.aoi:
        aoi = args.aoi
        event_date = args.event_date
    else:
        logger.info("No AOI specified; defaulting to Trishuli Valley case study.")
        aoi = TRISHULI_CASE_STUDY_BBOX
        event_date = TRISHULI_EVENT_DATE

    try:
        run_pipeline(aoi=aoi, event_date=event_date, output_dir=args.output, lang=args.lang)
    except Exception as e:
        logger.exception(f"Pipeline failed with unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
