"""Adversarial verifier ensuring zero ungrounded invented numbers in reports.

In compliance with Competition Rules §6.10:
Scans LLM or generated text for ungrounded digits or spelled-out numbers that do
not trace back to approved placeholders or dates.
"""

from __future__ import annotations

import re
from typing import Sequence

ADVERSARIAL_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ",
    "सात", "आठ", "नौ", "दश", "सय", "हजार",
]


class LedgerVerifier:
    """Rejects any text with numerals that are not explicitly whitelisted."""

    def __init__(self, whitelist_strings: Sequence[str] | None = None):
        self.whitelist: set[str] = set(whitelist_strings or [])

    def verify_template_compliance(self, text: str) -> tuple[bool, list[str]]:
        """Verify that template text before placeholder substitution contains no literal numerals."""
        violations = []

        # 1. Strip valid [[F:...]] placeholders so identifier numbers like km2 or run_01 don't trigger
        clean_text = re.sub(r"\[\[F:[^\]]+\]\]", " PLACEHOLDER ", text)

        # 2. Strip whitelisted dates or strings (e.g. 2026-08-26, 2026)
        for wl in sorted(self.whitelist, key=len, reverse=True):
            clean_text = clean_text.replace(wl, " WHITELIST ")

        # 3. Find remaining raw ASCII digits
        for match in re.finditer(r"\b\d+\b", clean_text):
            token = match.group(0)
            violations.append(f"Forbidden raw digit token '{token}' found in template at index {match.start()}")

        # 4. Find Devanagari digits
        for match in re.finditer(r"[०-९]+", clean_text):
            token = match.group(0)
            violations.append(f"Forbidden Devanagari numeral '{token}' found in template at index {match.start()}")

        # 5. Find suspicious spelled-out number words
        lower_text = clean_text.lower()
        for word in ADVERSARIAL_WORDS:
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, lower_text):
                violations.append(f"Suspicious spelled-out number word '{word}' found in unverified text")

        return len(violations) == 0, violations
