"""
Aggregate Factuality & Hallucination Metrics Module.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from factuality_benchmark.schema import FactualityResult
from factuality_benchmark.taxonomy import HallucinationType


class FactualityMetricsCalculator:
    """Calculates dataset-level factuality precision, recall, F1, and taxonomy distribution."""

    @staticmethod
    def calculate_aggregate_metrics(results: List[FactualityResult]) -> Dict[str, Any]:
        if not results:
            return {"total_items": 0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

        total = len(results)
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1_score for r in results]

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / total

        # Hallucination taxonomy breakdown
        taxonomy_counts: Dict[str, int] = {h.value: 0 for h in HallucinationType}
        for r in results:
            t_val = r.primary_hallucination_type.value if hasattr(r.primary_hallucination_type, "value") else str(r.primary_hallucination_type)
            taxonomy_counts[t_val] = taxonomy_counts.get(t_val, 0) + 1

        taxonomy_distribution = {
            t_type: {
                "count": count,
                "percentage": round((count / total) * 100, 2)
            }
            for t_type, count in taxonomy_counts.items() if count > 0
        }

        # Confusion statistics
        fp_count = sum(1 for r in results if not r.passed and r.precision > 0.5)
        fn_count = sum(1 for r in results if r.passed and r.precision < 1.0)

        return {
            "total_items": total,
            "passed_count": passed_count,
            "failed_count": total - passed_count,
            "pass_rate": round(pass_rate, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "taxonomy_distribution": taxonomy_distribution,
        }
