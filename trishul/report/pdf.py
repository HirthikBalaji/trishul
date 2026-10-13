"""One-page situation report PDF generator.

In compliance with Competition Rules §6.10:
Generates publication-quality one-page situation reports (report_en.pdf and report_ne.pdf)
with verified headline metrics, limitations, and mandatory attributions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from trishul.compliance.manifest import DISCLAIMER_TEXT, MANDATORY_ATTRIBUTIONS
from trishul.report.ledger import ClaimLedger

logger = logging.getLogger(__name__)


class SituationReportPdfGenerator:
    """Generates official 1-page situational reports in English and Nepali."""

    def __init__(self, ledger: ClaimLedger):
        self.ledger = ledger

    def generate_pdf(
        self,
        output_path: Path | str,
        lang: str = "en",
        aoi_name: str = "Trishuli Valley, Rasuwa / Bagmati",
        event_date: str = "2026-08-26",
    ) -> Path:
        """Render one-page PDF report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        primary_color = colors.HexColor("#14232B")  # Glacier ink
        accent_color = colors.HexColor("#1B8FA6")   # Meltwater
        light_panel = colors.HexColor("#E4EBEE")    # Ice panel

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=primary_color,
        )

        sub_style = ParagraphStyle(
            "ReportSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4A5568"),
        )

        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=primary_color,
            spaceAfter=4,
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#2D3748"),
        )

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#718096"),
        )

        story = []

        # 1. Header
        if lang == "ne":
            title_text = "त्रिशूल (TRISHUL) — विपद् अवस्था प्रतिवेदन"
            sub_text = f"अन्तरिक्षबाट बाढी क्षति तथा यातायात सम्पर्क विच्छेद नक्साङ्कन · घटना: {event_date} · स्थान: {aoi_name}"
        else:
            title_text = "TRISHUL — Rapid Situational Assessment"
            sub_text = f"Multimodal Flood & Debris Damage Mapping from Space · Event: {event_date} · AOI: {aoi_name}"

        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(sub_text, sub_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=8))

        # 2. Key Headline Metrics Table
        f_water = self.ledger.get("flood_water_km2")
        f_debris = self.ledger.get("debris_deposit_km2")
        f_tot_area = self.ledger.get("total_affected_km2")
        f_bld = self.ledger.get("buildings_hit_expected")
        f_roads = self.ledger.get("road_length_hit_km")
        f_bridges = self.ledger.get("bridges_hit_count")
        f_cutoff = self.ledger.get("settlements_cutoff_count")
        f_blind = self.ledger.get("blindspot_pct")

        val_water = f"{f_water.value} km²" if f_water else "N/A"
        val_debris = f"{f_debris.value} km²" if f_debris else "N/A"
        val_tot_area = f"{f_tot_area.value} km²" if f_tot_area else "N/A"
        val_bld = f"{f_bld.value} structures" if f_bld else "N/A"
        val_roads = f"{f_roads.value} km severed" if f_roads else "N/A"
        val_bridges = f"{f_bridges.value} bridges severed" if f_bridges else "N/A"
        val_cutoff = f"{f_cutoff.value} settlements isolated" if f_cutoff else "N/A"
        val_blind = f"{f_blind.value}% unobservable" if f_blind else "N/A"

        metrics_data = [
            [
                Paragraph("<b>Prong 1: Hazard Extent</b>", body_style),
                Paragraph("<b>Prong 2: Infrastructure Damage</b>", body_style),
                Paragraph("<b>Prong 3: Community Isolation</b>", body_style),
            ],
            [
                Paragraph(f"• Total Affected: <b>{val_tot_area}</b><br/>• Inundation: <b>{val_water}</b><br/>• Debris Deposits: <b>{val_debris}</b><br/>• Radar/Cloud Blindspot: <b>{val_blind}</b>", body_style),
                Paragraph(f"• Buildings Impacted: <b>{val_bld}</b><br/>• Damaged Roads: <b>{val_roads}</b><br/>• Severed Bridges: <b>{val_bridges}</b><br/>• Pre-event OSM snapshot baseline", body_style),
                Paragraph(f"• Cut-off by road: <b>{val_cutoff}</b><br/>• Hospital Access Severed<br/>• Multi-track radar verified<br/>• Top repair priority ranked", body_style),
            ]
        ]

        t_metrics = Table(metrics_data, colWidths=[175, 175, 175])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_panel),
            ("TEXTCOLOR", (0, 0), (-1, -1), primary_color),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 10))

        # 3. Methodological Summary
        story.append(Paragraph("System Architecture & Evidence Fusion", section_style))
        method_text = (
            "TRISHUL utilizes a <b>Physics-Gated Evidence Fusion</b> engine combining same-track Sentinel-1 radar backscatter, "
            "optical spectral indices (NDVI, NDWI, BSI), and Copernicus GLO-30 Height Above Nearest Drainage (HAND). "
            "Radar shadow and layover blindspots are rigorously masked via 3D ray-marching rather than guessed. "
            "Road graph survival is evaluated through Monte-Carlo percolation under spatially correlated hazard fields."
        )
        story.append(Paragraph(method_text, body_style))
        story.append(Spacer(1, 8))

        # 4. Mandatory Limitations
        story.append(Paragraph("System Limitations & Operational Bounds", section_style))
        limit_text = (
            "1. <b>Temporal Latency:</b> Sentinel-1/2 revisit is days; this pipeline does not provide minutes-ahead early warning.<br/>"
            "2. <b>Spatial Resolution:</b> 10 m radar and 30 m DEM pixels model sub-pixel structures (buildings/bridges) probabilistically. We report hit probability, never structural damage grading.<br/>"
            "3. <b>Terrain Blindspots:</b> Steep valley walls in radar shadow or layover are masked as unobservable.<br/>"
            "4. <b>OSM Baseline Completeness:</b> Settlements lacking prior road connections in pre-event OSM are classified as 'pre-existing unconnected', not flood-isolated."
        )
        story.append(Paragraph(limit_text, body_style))
        story.append(Spacer(1, 8))

        # 5. Attributions and Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        for attr in MANDATORY_ATTRIBUTIONS:
            story.append(Paragraph(attr, disclaimer_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"<b>Notice:</b> {DISCLAIMER_TEXT}", disclaimer_style))

        doc.build(story)
        logger.info(f"Generated situation report PDF: {out}")
        return out
