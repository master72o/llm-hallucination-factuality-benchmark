"""
Unit tests for factuality benchmark data schemas.
"""

import pytest
from factuality_benchmark.taxonomy import HallucinationType, ClaimStatus
from factuality_benchmark.schema import AtomicClaim, FactualityItem, FactualityResult


def test_hallucination_taxonomy_enum():
    assert HallucinationType.CORRECT_CLAIM.value == "correct_claim"
    assert HallucinationType.FABRICATED_ENTITY.value == "fabricated_entity"
    assert HallucinationType.INCORRECT_DATE.value == "incorrect_date"


def test_atomic_claim_serialization():
    c = AtomicClaim(
        id="c1",
        text="Ottawa is capital of Canada",
        status=ClaimStatus.SUPPORTED,
        hallucination_type=HallucinationType.CORRECT_CLAIM,
        explanation="Supported by evidence"
    )
    d = c.to_dict()
    assert d["id"] == "c1"
    assert d["status"] == "supported"
    assert d["hallucination_type"] == "correct_claim"


def test_factuality_item_deserialization():
    raw = {
        "id": "f1",
        "prompt": "Capital of Canada?",
        "response": "Ottawa",
        "evidence_text": "Ottawa is capital of Canada",
        "expected_hallucination_type": "correct_claim",
        "domain": "Geography"
    }
    item = FactualityItem.from_dict(raw)
    assert item.id == "f1"
    assert item.expected_hallucination_type == HallucinationType.CORRECT_CLAIM
