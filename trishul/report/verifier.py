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
        # Find raw ASCII digits
        for match in re.finditer(r"\b\d+\b", text):
            token = match.group(0)
            if token not in self.whitelist:
                violations.append(f"Forbidden raw digit token '{token}' found in template at index {match.start()}")

        # Find Devanagari digits
        for match in re.finditer(r"[०-९]+", text):
            token = match.group(0)
            if token not in self.whitelist:
                violations.append(f"Forbidden Devanagari numeral '{token}' found in template at index {match.start()}")

        # Find suspicious spelled-out number words
        lower_text = text.lower()
        for word in ADVERSARIAL_WORDS:
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, lower_text):
                violations.append(f"Suspicious spelled-out number word '{word}' found in unverified text")

        return len(violations) == 0, violations
