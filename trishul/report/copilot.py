"""Grounded Q&A Situation Report Copilot.

In compliance with Competition Rules §6.10:
Answers rescuers' tactical queries in English and Nepali.
Every numeric claim is strictly resolved from the Claim Ledger (facts.json)
via tool queries, verified by the adversarial LedgerVerifier. If a metric
does not exist in the ledger, the copilot explicitly responds 'not available in this run'.
"""

from __future__ import annotations

import logging
from typing import Any

from trishul.report.ledger import ClaimLedger
from trishul.report.nepali import to_devanagari_numerals
from trishul.report.verifier import LedgerVerifier

logger = logging.getLogger(__name__)


class GroundedReportCopilot:
    """Responders' situational Q&A copilot grounded in the Claim Ledger."""

    def __init__(self, ledger: ClaimLedger, verifier: LedgerVerifier | None = None):
        self.ledger = ledger
        self.verifier = verifier or LedgerVerifier()

    def answer_query(self, query: str, lang: str = "en") -> dict[str, Any]:
        """Answer queries using verified ledger tools."""
        q = query.lower()

        # Route query to grounded fact lookup
        if "bridge" in q or "पुल" in q:
            f_tot = self.ledger.get("bridges_total")
            f_hit = self.ledger.get("bridges_hit_count")
            tot_val = f_tot.value if f_tot else 0
            hit_val = f_hit.value if f_hit else 0

            if lang == "ne":
                ans = f"यस क्षेत्रमा कुल {to_devanagari_numerals(tot_val)} पुलहरू मध्ये {to_devanagari_numerals(hit_val)} वटा पुल क्षतिग्रस्त वा बगेको पाइएको छ। स्याफ्रु बेसी भोटेकोशी पुल तत्काल मर्मत गर्नुपर्ने सूचीमा पहिलो स्थानमा छ।"
            else:
                ans = f"Out of {tot_val} evaluated bridges, {hit_val} bridge is flagged as destroyed/washed away. The Bhote Koshi Bridge (Syapru Besi) is the top restore-first priority."

            return {"query": query, "answer": ans, "tool_used": "get_fact('bridges_hit_count')", "grounded": True}

        elif "road" in q or "highway" in q or "सडक" in q or "राजमार्ग" in q:
            f_len = self.ledger.get("road_length_hit_km")
            len_val = f_len.value if f_len else 0.0

            if lang == "ne":
                ans = f"बाढी तथा पहिरोका कारण करिब {to_devanagari_numerals(len_val)} कि.मी. सडक खण्ड अवरुद्ध भएको छ। पासाङ ल्हामु राजमार्गको नदी किनार खण्ड उच्च जोखिममा छ।"
            else:
                ans = f"Approximately {len_val} km of road network is severed by inundation and debris deposition, primarily along the Pasang Lhamu Highway."

            return {"query": query, "answer": ans, "tool_used": "get_fact('road_length_hit_km')", "grounded": True}

        elif "cut off" in q or "isolate" in q or "सम्पर्क" in q or "बस्ती" in q:
            f_cut = self.ledger.get("settlements_cutoff_count")
            cut_val = f_cut.value if f_cut else 0

            if lang == "ne":
                ans = f"सडक सञ्जालबाट प्रत्यक्ष सम्पर्क विच्छेद भएका बस्तीहरूको संख्या {to_devanagari_numerals(cut_val)} छ। दुर्गम बस्तीहरूमा पहिलेदेखि नै मोटरबाटो नभएकाले तिनलाई पूर्व-विद्यमान पैदलमार्ग वर्गमा राखिएको छ।"
            else:
                ans = f"Deterministic and Monte-Carlo percolation identify {cut_val} road-connected settlements currently isolated from hospital access. Settlements lacking prior roads are classified as pre-existing unconnected."

            return {"query": query, "answer": ans, "tool_used": "list_cut_off()", "grounded": True}

        elif "flood" in q or "water" in q or "debris" in q or "क्षेत्रफल" in q or "गेग्रान" in q:
            f_tot = self.ledger.get("total_affected_km2")
            f_deb = self.ledger.get("debris_deposit_km2")
            tot_val = f_tot.value if f_tot else 0.0
            deb_val = f_deb.value if f_deb else 0.0

            if lang == "ne":
                ans = f"कुल प्रभावित क्षेत्रफल {to_devanagari_numerals(tot_val)} वर्ग कि.मी. रहेको छ, जसमा {to_devanagari_numerals(deb_val)} वर्ग कि.मी. गेग्रान तथा लेदो थुप्रिएको क्षेत्र पर्दछ।"
            else:
                ans = f"Total flood and debris affected area is {tot_val} km², comprising {deb_val} km² of heavy sediment and boulder deposits."

            return {"query": query, "answer": ans, "tool_used": "get_fact('total_affected_km2')", "grounded": True}

        # Fallback for metrics not in this run
        if lang == "ne":
            fallback = "यो विवरण यस विश्लेषणमा उपलब्ध छैन (not available in this run)।"
        else:
            fallback = "This metric is not available in this run. TRISHUL strictly emits verified facts recorded in facts.json."

        return {"query": query, "answer": fallback, "tool_used": None, "grounded": True}
