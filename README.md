# LLM Hallucination & Factuality Benchmark

[![CI Pipeline](https://github.com/master72o/llm-hallucination-factuality-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/llm-hallucination-factuality-benchmark/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A claim-level evidence verification framework and benchmark evaluating LLM factuality, precision, recall, F1, and hallucination taxonomy categorization (*Fabricated Entities, Incorrect Dates, Incorrect Numbers, False Attributions, Contradictions*).

---

## Overview

Large Language Models frequently produce fluent but demonstrably false assertions ("hallucinations"). Measuring overall string similarity (e.g. ROUGE, BLEU) fails to catch subtle numerical, chronological, or entity fabrications. **LLM Hallucination & Factuality Benchmark** decomposes candidate completions into sentence-level atomic claims, verifies each claim against reference evidence documents, and calculates claim-level Precision, Recall, and F1 scores.

---

## Problem Statement

Evaluating factuality in generative AI models introduces three key engineering challenges:
1. **Coarse-Grained Metrics**: Traditional n-gram overlap metrics treat all words equally, failing to penalize critical factual errors (e.g. writing $2026$ instead of $1789$).
2. **Lack of Claim Decomposition**: Full-paragraph evaluations conflate valid background context with isolated false assertions.
3. **Unstructured Error Diagnosis**: Automated evaluation tools rarely classify *why* a model hallucinated (e.g. date mismatch vs entity fabrication vs direct contradiction).

---

## Objective

Build a reusable, automated factuality benchmark that:
- Decomposes responses into **atomic sentence-level claims**.
- Compares claims against ground-truth reference evidence documents.
- Categorizes hallucinations into an 8-category **Hallucination Taxonomy**.
- Computes claim-level **Macro & Micro Precision**, **Recall**, and **F1 Score**.
- Exports structured evaluation reports (`JSON`, `CSV`, Markdown) and visualization charts.

---

## Research Questions

1. *How does sentence-level claim decomposition improve hallucination detection sensitivity compared to full-text overlap metrics?*
2. *Which hallucination categories (e.g. incorrect numbers vs false attributions) occur with the highest frequency in ungrounded LLM completions?*
3. *What threshold of Precision ($\ge 0.80$) effectively guarantees evidence faithfulness in production RAG systems?*

---

## Why This Matters

Uncaught hallucinations degrade user trust and present severe liability in critical domains (finance, healthcare, legal, technical assistance). Automated claim verification guarantees that model outputs remain strictly grounded in verified facts.

---

## Architecture

```
                                +-----------------------------+
                                |  Candidate Response &       |
                                |  Evidence Text Document     |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Atomic Claim Extractor      |
                                | (Sentence-level Regex)      |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Evidence Verification Engine|
                                | - Numerical & Date Checker  |
                                | - Entity & Relation Matcher |
                                | - Contradiction Classifier  |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Hallucination Taxonomy      |
                                | Classification & Metrics    |
                                | - Precision, Recall, F1      |
                                +--------------+--------------+
                                               |
                 +----------------------+------+----------------------+
                 |                             |                      |
                 v                             v                      v
      +----------+----------+        +---------+----------+ +---------+----------+
      | JSON / CSV Exports  |        | Markdown Summary   | | Matplotlib Figures |
      | (results/ directory)|        | Report (reports/)  | | (reports/figures/) |
      +---------------------+        +--------------------+ +--------------------+
```

---

## Dataset

Evaluations are conducted on `data/factuality_benchmark_dataset.jsonl`, a 20-item benchmark dataset containing verified reference evidence texts across History, Physics, Astronomy, Chemistry, Computer Science, Biology, and Geography.

Full data governance is in [`data/README.md`](data/README.md).

---

## Data Collection / Construction

- **Benchmark Pairing**: Each prompt item is paired with a candidate LLM completion and a verified reference evidence document.
- **Controlled Negative Test Vectors**: Test items include specific synthetic hallucination categories (`incorrect_date`, `incorrect_number`, `false_attribution`, `contradiction`, `fabricated_entity`).

---

## Annotation Guidelines

1. **Claim Extraction**: Split response into independent declarative sentences.
2. **Evidence Verification**: A claim is `SUPPORTED` if its core factual assertion is directly entailed by the evidence document.
3. **Factuality Precision**:
$$\text{Precision} = \frac{\text{Supported Claims}}{\text{Total Claims Evaluated}}$$

---

## Evaluation Rubric

- `SUPPORTED`: Claim is fully backed by reference evidence.
- `UNSUPPORTED`: Claim assertion cannot be verified from provided evidence.
- `CONTRADICTION`: Claim directly contradicts reference evidence.

---

## Error Taxonomy

Hallucinations are classified into 8 discrete categories:

1. `correct_claim`: Fully supported factual statement.
2. `fabricated_entity`: Non-existent person, fake URL, or fake organization.
3. `incorrect_date`: Wrong year, century, or chronological sequence.
4. `incorrect_number`: Wrong calculation, statistic, or quantitative figure.
5. `false_attribution`: Assigning an action/invention to the wrong entity.
6. `unsupported_claim`: Unverified assertion.
7. `contradiction`: Direct negation of verified evidence.
8. `partially_supported`: Mixed true and unverified assertions.

---

## Metrics

- **Macro Precision**: Mean proportion of factual claims ($[0.0, 1.0]$).
- **Macro Recall**: Mean proportion of actual hallucinations detected.
- **Macro F1 Score**: Harmonic mean of Precision and Recall.
- **Pass Rate**: Percentage of items with Precision $\ge 0.80$ and no `CONTRADICTION`.

---

## Experimental Design

The benchmark pipeline extracts atomic claims, compares entities and numbers against evidence documents, computes precision/recall/F1, and aggregates taxonomy statistics.

---

## Installation

```bash
# Clone repository
git clone https://github.com/master72o/llm-hallucination-factuality-benchmark.git
cd llm-hallucination-factuality-benchmark

# Activate virtual environment
source ../.venv/bin/activate

# Install package in editable mode
pip install -e .
```

---

## Usage

### Run Factuality Benchmark CLI
```bash
python -m factuality_benchmark.cli evaluate \
  --input data/factuality_benchmark_dataset.jsonl \
  --output-dir results/ \
  --report reports/summary_report.md \
  --figures-dir reports/figures
```

### Run Pytest Suite
```bash
pytest --cov=factuality_benchmark tests/
```

---

## Example

```python
from factuality_benchmark.schema import FactualityItem
from factuality_benchmark.factuality_checker import FactualityChecker

item = FactualityItem(
    id="example_01",
    prompt="When was Python released?",
    response="Python was released in 1999.",
    evidence_text="Python was first released on February 20, 1991."
)

checker = FactualityChecker()
result = checker.evaluate_item(item)

print(f"Precision: {result.precision}")
print(f"Primary Hallucination: {result.primary_hallucination_type.value}")
```

---

## Results

Empirical evaluation on the 20-item benchmark dataset yielded the following measured results:

- **Total Evaluated Items**: `20`
- **Passed Quality Gate (Precision >= 0.80 & No Critical Error)**: `12` (`60.0%`)
- **Failed Quality Gate**: `8` (`40.0%`)
- **Macro Precision**: `0.6000`
- **Macro Recall**: `1.0000`
- **Macro F1 Score**: `0.7333`

### Hallucination Taxonomy Distribution

| Category | Count | Percentage |
| :--- | :--- | :--- |
| `correct_claim` | `12` | `60.0%` |
| `incorrect_number` | `3` | `15.0%` |
| `false_attribution` | `2` | `10.0%` |
| `contradiction` | `2` | `10.0%` |
| `incorrect_date` | `1` | `5.0%` |

*Full research summary report available in [`reports/summary_report.md`](reports/summary_report.md).*

---

## Failure Analysis

The benchmark detected 8 failure instances:
1. **Chronological Error (`fact_04`)**: Stated French Revolution began in 1848 instead of 1789 (`incorrect_date`).
2. **Mathematical Error (`fact_05`)**: Calculated 15 * 14 = 225 instead of 210 (`incorrect_number`).
3. **Attribution Errors (`fact_06`, `fact_14`)**: Assigned Mona Lisa to Van Gogh and telephone to Edison (`false_attribution`).
4. **Contradictions (`fact_18`, `fact_20`)**: Stated Atlantic is largest ocean and Spanish is official language of Brazil (`contradiction`).

---

## Limitations

- **Syntactic Parser**: Claim extraction currently relies on sentence-level punctuation regex.
- **Language Scope**: Benchmark rules are tailored for English reference evidence.

---

## Ethical / Safety Considerations

- All benchmark evidence texts use public domain scientific and historical facts.

---

## Reproducibility

1. Activate virtual environment: `source ../.venv/bin/activate`
2. Run `pytest` to confirm test suite integrity.
3. Run `python -m factuality_benchmark.cli evaluate --input data/factuality_benchmark_dataset.jsonl --output-dir results/ --report reports/summary_report.md`
4. Inspect figures in `reports/figures/`.

---

## Future Improvements

- Add NLI entailment model integration (e.g. `DeBERTa-v3-large-mnli`).
- Implement coreference resolution for complex multi-sentence completions.

---

## References

- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. arXiv:2305.14251.
- Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys.

---

## Author

**LLM Factuality Specialist & AI Evaluator**  
*Specializing in Claim Verification, Evidence Grounding, and Hallucination Taxonomy Analysis.*

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in data/.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run pytest or the package CLI to re-compute all scores, metrics, and summary tables locally.
