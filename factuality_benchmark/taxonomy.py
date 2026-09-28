"""
Hallucination Taxonomy & Claim Verification Definitions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class HallucinationType(str, Enum):
    CORRECT_CLAIM = "correct_claim"
    FABRICATED_ENTITY = "fabricated_entity"
    INCORRECT_DATE = "incorrect_date"
    INCORRECT_NUMBER = "incorrect_number"
    FALSE_ATTRIBUTION = "false_attribution"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    PARTIALLY_SUPPORTED = "partially_supported"


@dataclass
class TaxonomyDetail:
    category: HallucinationType
    name: str
    description: str
    severity: str
    detection_heuristic: str


HALLUCINATION_TAXONOMY: Dict[HallucinationType, TaxonomyDetail] = {
    HallucinationType.CORRECT_CLAIM: TaxonomyDetail(
        category=HallucinationType.CORRECT_CLAIM,
        name="Correct Factual Claim",
        description="Claim is fully supported by verified reference evidence.",
        severity="none",
        detection_heuristic="Exact match or verified entailment with evidence document."
    ),
    HallucinationType.FABRICATED_ENTITY: TaxonomyDetail(
        category=HallucinationType.FABRICATED_ENTITY,
        name="Fabricated Entity",
        description="Claim invents non-existent people, fake URLs, non-existent papers, or fake organizations.",
        severity="high",
        detection_heuristic="Entity verification against knowledge graph or reference text."
    ),
    HallucinationType.INCORRECT_DATE: TaxonomyDetail(
        category=HallucinationType.INCORRECT_DATE,
        name="Incorrect Date / Chronology",
        description="Claim contains a wrong year, century, or chronological sequence.",
        severity="high",
        detection_heuristic="Numerical date comparison against reference evidence."
    ),
    HallucinationType.INCORRECT_NUMBER: TaxonomyDetail(
        category=HallucinationType.INCORRECT_NUMBER,
        name="Incorrect Numerical Value",
        description="Claim contains a wrong calculation result, wrong statistic, or inaccurate quantitative value.",
        severity="high",
        detection_heuristic="Numerical extraction and range/value checking."
    ),
    HallucinationType.FALSE_ATTRIBUTION: TaxonomyDetail(
        category=HallucinationType.FALSE_ATTRIBUTION,
        name="False Attribution",
        description="Claim assigns an action, invention, or quote to the wrong person or entity.",
        severity="high",
        detection_heuristic="Subject-predicate relation matching against reference evidence."
    ),
    HallucinationType.UNSUPPORTED_CLAIM: TaxonomyDetail(
        category=HallucinationType.UNSUPPORTED_CLAIM,
        name="Unsupported Claim",
        description="Claim makes an assertion that cannot be verified or derived from provided reference evidence.",
        severity="medium",
        detection_heuristic="Evidence entailment check returns neutral or zero support."
    ),
    HallucinationType.CONTRADICTION: TaxonomyDetail(
        category=HallucinationType.CONTRADICTION,
        name="Direct Contradiction",
        description="Claim directly states the opposite of verified reference evidence.",
        severity="critical",
        detection_heuristic="Contradiction classifier or negated entity assertion."
    ),
    HallucinationType.PARTIALLY_SUPPORTED: TaxonomyDetail(
        category=HallucinationType.PARTIALLY_SUPPORTED,
        name="Partially Supported Claim",
        description="Claim mixes accurate statements with unverified or slightly altered facts.",
        severity="medium",
        detection_heuristic="Partial token/entity overlap with reference evidence."
    ),
}
