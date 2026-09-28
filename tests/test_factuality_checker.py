"""
Unit tests for Factuality Checker & Claim Verification Engine.
"""

import pytest
from factuality_benchmark.schema import FactualityItem
from factuality_benchmark.taxonomy import HallucinationType
from factuality_benchmark.factuality_checker import FactualityChecker


@pytest.fixture
def checker():
    return FactualityChecker()


def test_factuality_checker_pass(checker):
    item = FactualityItem(
        id="f1",
        prompt="Capital of Canada?",
        response="Ottawa is the capital of Canada.",
        evidence_text="Ottawa is the official capital of Canada.",
        expected_hallucination_type=HallucinationType.CORRECT_CLAIM
    )
    result = checker.evaluate_item(item)
    assert result.precision >= 0.8
    assert result.passed is True


def test_factuality_checker_number_mismatch(checker):
    item = FactualityItem(
        id="f2",
        prompt="Math 15*14",
        response="15 multiplied by 14 equals 225.",
        evidence_text="15 multiplied by 14 is equal to 210.",
        expected_hallucination_type=HallucinationType.INCORRECT_NUMBER
    )
    result = checker.evaluate_item(item)
    assert result.passed is False
    assert result.primary_hallucination_type == HallucinationType.INCORRECT_NUMBER
