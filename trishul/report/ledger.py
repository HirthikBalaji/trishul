"""Claim Ledger (facts.json) and template substitution engine.

In compliance with Competition Rules §0 Rule 4 and §6.10:
Every single metric, figure, and quantity originates from facts.json and is
injected automatically via [[F:id]] placeholders. No numbers are invented.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from trishul.report.nepali import to_devanagari_numerals

logger = logging.getLogger(__name__)


@dataclass
class FactRecord:
    id: str
    value: float | int | str
    unit: str
    ci_lo: float | None = None
    ci_hi: float | None = None
    layer: str = "general"
    method: str = "direct"
    aoi: str = "aoi"
    run_id: str = "run_01"


class ClaimLedger:
    """Central facts ledger storing auditable, script-produced quantities."""

    def __init__(self, run_id: str = "run_01"):
        self.run_id = run_id
        self._facts: dict[str, FactRecord] = {}

    def register(
        self,
        fact_id: str,
        value: float | int | str,
        unit: str,
        ci_lo: float | None = None,
        ci_hi: float | None = None,
        layer: str = "general",
        method: str = "direct",
    ) -> FactRecord:
        """Register a single verifiable fact into the ledger."""
        rec = FactRecord(
            id=fact_id,
            value=value,
            unit=unit,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            layer=layer,
            method=method,
            aoi="aoi",
            run_id=self.run_id,
        )
        self._facts[fact_id] = rec
        return rec

    def get(self, fact_id: str) -> FactRecord | None:
        return self._facts.get(fact_id)

    def export_json(self, output_dir: Path | str) -> Path:
        """Write facts.json to run output directory."""
        out_path = Path(output_dir) / "facts.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(rec) for rec in self._facts.values()]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path

    def render_text(self, template_str: str, lang: str = "en") -> str:
        """Replace [[F:id]] placeholders with localized, formatted fact values."""
        def repl(match: re.Match[str]) -> str:
            fid = match.group(1).strip()
            rec = self._facts.get(fid)
            if not rec:
                return f"[MISSING_FACT:{fid}]"

            val_str = str(rec.value)
            if rec.ci_lo is not None and rec.ci_hi is not None:
                val_str = f"{rec.value} (90% CI: {rec.ci_lo}–{rec.ci_hi})"

            if rec.unit:
                val_str = f"{val_str} {rec.unit}".strip()

            if lang == "ne":
                val_str = to_devanagari_numerals(val_str)

            return val_str

        return re.sub(r"\[\[F:([^\]]+)\]\]", repl, template_str)
