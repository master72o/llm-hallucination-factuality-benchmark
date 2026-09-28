"""
Unit tests for Factuality Metrics Calculator.
"""

import pytest
from factuality_benchmark.schema import FactualityResult, AtomicClaim, ClaimStatus
from factuality_benchmark.taxonomy import HallucinationType
from factuality_benchmark.metrics import FactualityMetricsCalculator


def test_calculate_aggregate_metrics():
    res1 = FactualityResult(
        item_id="f1", prompt="p", response="r", evidence_text="e",
        claims=[], precision=1.0, recall=1.0, f1_score=1.0,
        primary_hallucination_type=HallucinationType.CORRECT_CLAIM, passed=True
    )
    res2 = FactualityResult(
        item_id="f2", prompt="p", response="r", evidence_text="e",
        claims=[], precision=0.0, recall=0.0, f1_score=0.0,
        primary_hallucination_type=HallucinationType.INCORRECT_NUMBER, passed=False
    )

    metrics = FactualityMetricsCalculator.calculate_aggregate_metrics([res1, res2])
    assert metrics["total_items"] == 2
    assert metrics["macro_precision"] == 0.5
    assert metrics["pass_rate"] == 0.5
    assert "incorrect_number" in metrics["taxonomy_distribution"]
