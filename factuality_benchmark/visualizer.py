"""
Visualization Generator for LLM Hallucination & Factuality Benchmark.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class Visualizer:
    """Generates charts for factuality benchmark metrics and hallucination taxonomy."""

    @staticmethod
    def generate_all_figures(metrics: Dict[str, Any], output_dir: str = "reports/figures") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Factuality Metrics Plot
        met_path = os.path.join(output_dir, "factuality_metrics.png")
        Visualizer._plot_factuality_metrics(metrics, met_path)
        generated["factuality_metrics"] = met_path

        # 2. Hallucination Taxonomy Breakdown Plot
        tax_path = os.path.join(output_dir, "hallucination_taxonomy_breakdown.png")
        Visualizer._plot_taxonomy_breakdown(metrics.get("taxonomy_distribution", {}), tax_path)
        generated["hallucination_taxonomy_breakdown"] = tax_path

        return generated

    @staticmethod
    def _plot_factuality_metrics(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["Precision", "Recall", "F1 Score", "Pass Rate"]
        vals = [
            metrics.get("macro_precision", 0.0),
            metrics.get("macro_recall", 0.0),
            metrics.get("macro_f1", 0.0),
            metrics.get("pass_rate", 0.0)
        ]
        colors = ["#2b5c8f", "#5bc0de", "#5cb85c", "#f0ad4e"]

        bars = ax.bar(labels, vals, color=colors, edgecolor="#333333", width=0.45)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Score / Rate", fontsize=11, fontweight="bold")
        ax.set_title("Factuality & Evidence Grounding Performance Metrics", fontsize=13, fontweight="bold", pad=15)
        ax.axhline(y=0.80, color="#d9534f", linestyle="--", linewidth=1.2, label="Target Quality Threshold (0.80)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_taxonomy_breakdown(tax_dist: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))

        if not tax_dist:
            ax.text(0.5, 0.5, "No Hallucinations Detected", ha="center", va="center", fontsize=12)
        else:
            categories = list(tax_dist.keys())
            counts = [tax_dist[c]["count"] for c in categories]

            bars = ax.barh(categories, counts, color="#d9534f", edgecolor="#a94442", height=0.5)
            ax.set_xlabel("Frequency Count", fontsize=11, fontweight="bold")
            ax.set_title("Hallucination Taxonomy Frequency Breakdown", fontsize=13, fontweight="bold", pad=15)
            ax.grid(axis="x", linestyle=":", alpha=0.6)

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", ha="left", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
