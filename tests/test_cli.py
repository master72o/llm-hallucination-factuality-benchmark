"""
Integration test for factuality benchmark CLI.
"""

import os
import pytest
from factuality_benchmark.cli import load_factuality_dataset
from factuality_benchmark.factuality_checker import FactualityChecker
from factuality_benchmark.metrics import FactualityMetricsCalculator
from factuality_benchmark.report_generator import ReportGenerator


def test_cli_load_dataset(tmp_path):
    jsonl = tmp_path / "fact_test.jsonl"
    jsonl.write_text(
        '{"id": "f1", "prompt": "Hi", "response": "Hello", "evidence_text": "Hello"}\n',
        encoding="utf-8"
    )

    items = load_factuality_dataset(str(jsonl))
    assert len(items) == 1
    assert items[0].id == "f1"


def test_full_factuality_pipeline_execution(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"
    dataset_file.write_text(
        '{"id": "f1", "prompt": "Hi", "response": "Hello", "evidence_text": "Hello"}\n',
        encoding="utf-8"
    )

    items = load_factuality_dataset(str(dataset_file))
    checker = FactualityChecker()
    results = [checker.evaluate_item(items[0])]
    metrics = FactualityMetricsCalculator.calculate_aggregate_metrics(results)

    out_dir = tmp_path / "results"
    report_path = tmp_path / "report.md"

    ReportGenerator.export_results(results, str(out_dir))
    ReportGenerator.generate_markdown_report(metrics, str(report_path))

    assert os.path.exists(out_dir / "factuality_results.json")
    assert os.path.exists(out_dir / "factuality_results.csv")
    assert os.path.exists(report_path)
