"""
Command-Line Interface for LLM Hallucination & Factuality Benchmark.
"""

import os
import json
import argparse
from typing import List
from factuality_benchmark.schema import FactualityItem, FactualityResult
from factuality_benchmark.factuality_checker import FactualityChecker
from factuality_benchmark.metrics import FactualityMetricsCalculator
from factuality_benchmark.report_generator import ReportGenerator
from factuality_benchmark.visualizer import Visualizer


def load_factuality_dataset(file_path: str) -> List[FactualityItem]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    items: List[FactualityItem] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(FactualityItem.from_dict(data))
        else:
            data_list = json.load(f)
            for data in data_list:
                items.append(FactualityItem.from_dict(data))
    return items


def main():
    parser = argparse.ArgumentParser(description="LLM Hallucination & Factuality Benchmark CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate dataset factuality against evidence")
    eval_parser.add_argument("--input", "-i", required=True, help="Path to factuality benchmark dataset (.jsonl)")
    eval_parser.add_argument("--output-dir", "-o", default="results", help="Directory to save output JSON/CSV")
    eval_parser.add_argument("--report", "-r", default="reports/summary_report.md", help="Path to generate report")
    eval_parser.add_argument("--figures-dir", "-f", default="reports/figures", help="Directory to save figures")

    args = parser.parse_args()

    if args.command == "evaluate":
        print(f"Loading factuality benchmark dataset from: {args.input}")
        items = load_factuality_dataset(args.input)
        print(f"Loaded {len(items)} evaluation items.")

        print("Executing claim-level evidence verification and taxonomy classification...")
        checker = FactualityChecker()
        results: List[FactualityResult] = []

        for idx, item in enumerate(items, 1):
            res = checker.evaluate_item(item)
            results.append(res)
            print(f" [{idx}/{len(items)}] ID: {item.id} | Precision: {res.precision:.2f} | Passed: {res.passed} | Taxonomy: {res.primary_hallucination_type.value}")

        print("\nCalculating aggregate factuality metrics (Macro Precision, Recall, F1)...")
        metrics = FactualityMetricsCalculator.calculate_aggregate_metrics(results)
        print(f" Macro Precision: {metrics['macro_precision']:.4f}")
        print(f" Macro Recall: {metrics['macro_recall']:.4f}")
        print(f" Macro F1 Score: {metrics['macro_f1']:.4f}")

        print("\nExporting results to JSON and CSV...")
        ReportGenerator.export_results(results, args.output_dir)

        print("\nGenerating visual report figures...")
        Visualizer.generate_all_figures(metrics, args.figures_dir)

        print(f"\nGenerating Markdown summary report at: {args.report}")
        ReportGenerator.generate_markdown_report(metrics, args.report)

        print("\nFactuality Benchmark Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()
