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


def test_factuality_checker_date_mismatch(checker):
    item = FactualityItem(
        id="f3",
        prompt="When was Python 3.0 released?",
        response="Python 3.0 was released in 2012.",
        evidence_text="Python 3.0 was officially released in 2008.",
        expected_hallucination_type=HallucinationType.INCORRECT_DATE,
    )
    result = checker.evaluate_item(item)
    assert result.passed is False
    assert result.primary_hallucination_type == HallucinationType.INCORRECT_DATE


def test_claim_density_and_risk_profile(checker):
    item_ok = FactualityItem(
        id="f4",
        prompt="Capital of Canada?",
        response="Ottawa is the capital of Canada.",
        evidence_text="Ottawa is the official capital of Canada.",
        expected_hallucination_type=HallucinationType.CORRECT_CLAIM,
    )
    res_ok = checker.evaluate_item(item_ok)
    profile_ok = checker.compute_claim_density_and_risk_profile(res_ok)
    assert profile_ok["risk_tier"] == "LOW_RISK"
    assert profile_ok["contradicted_claim_rate"] == 0.0
    assert profile_ok["claims_per_100_words"] > 0.0

    item_bad = FactualityItem(
        id="f5",
        prompt="Math 15*14",
        response="15 multiplied by 14 equals 225.",
        evidence_text="15 multiplied by 14 is equal to 210.",
        expected_hallucination_type=HallucinationType.INCORRECT_NUMBER,
    )
    res_bad = checker.evaluate_item(item_bad)
    profile_bad = checker.compute_claim_density_and_risk_profile(res_bad)
    assert profile_bad["risk_tier"] == "CRITICAL_RISK"
    assert profile_bad["contradicted_claim_rate"] > 0.0
