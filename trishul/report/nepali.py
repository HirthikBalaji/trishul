"""Nepali linguistic utilities and disaster domain terminology dictionary.

In compliance with Competition Rules §6.10:
Provides curated translations of flood and damage terms, Devanagari numeral
conversion, and localization formatting.
"""

from __future__ import annotations

# Devanagari digit mapping
DEVANAGARI_DIGITS = {
    "0": "०",
    "1": "१",
    "2": "२",
    "3": "३",
    "4": "४",
    "5": "५",
    "6": "६",
    "7": "७",
    "8": "८",
    "9": "९",
}


def to_devanagari_numerals(number_str: str | int | float) -> str:
    """Convert ASCII Arabic digits to Devanagari numeral glyphs."""
    s = str(number_str)
    return "".join(DEVANAGARI_DIGITS.get(ch, ch) for ch in s)


# Domain terminology dictionary
NEPALI_GLOSSARY = {
    "situation_report": "विपद् अवस्था प्रतिवेदन",
    "mission": "त्रिशूल - अन्तरिक्षबाट बाढी क्षति नक्साङ्कन",
    "event_date": "घटना मिति",
    "flood_affected_area": "बाढी तथा गेग्रान प्रभावित क्षेत्रफल",
    "flood_water": "जलमग्न क्षेत्र (बाढीको पानी)",
    "debris_deposit": "गेग्रान तथा लेदो थुप्रिएको क्षेत्र",
    "scour_widening": "नदी कटान तथा चौडा भएको भाग",
    "buildings_impacted": "प्रभावित घर/संरचना संख्या",
    "roads_severed": "अवरुद्ध सडक लम्बाइ",
    "bridges_destroyed": "क्षतिग्रस्त/बगेका पुलहरू",
    "cut_off_settlements": "सम्पर्क विच्छेद भएका बस्तीहरू",
    "reachable_settlements": "सडक सम्पर्क यथावत रहेका बस्तीहरू",
    "blindspot_area": "राडार तथा बादलका कारण नदेखिएको क्षेत्र (ब्लाइन्डस्पट)",
    "not_assessable": "मूल्याङ्कन गर्न नसकिएको",
    "prototype_disclaimer": "यो शैक्षिक प्रतिस्पर्धात्मक नमुना हो, आपत्कालीन उद्धारको आधिकारिक निर्णयमा प्रयोग नगर्नुहोला।",
}
