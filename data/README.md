# Factuality Benchmark Data Governance & Experiment Setup

## Dataset Metadata

- **Name**: LLM Factuality & Evidence Grounding Benchmark (`factuality_benchmark_dataset.jsonl`)
- **Version**: 1.0.0
- **Format**: JSON Lines (`.jsonl`)
- **Sample Count**: 20 benchmark test instances
- **Domains Covered**: History, Physics, Astronomy, Chemistry, Computer Science, Biology, Geography.
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

## Intended Use & Methodology

This benchmark measures an LLM's ability to remain faithful to reference evidence documents. Each record contains:
- `id`: Unique identifier.
- `prompt`: User prompt query.
- `response`: Candidate LLM response.
- `evidence_text`: Verified reference document containing true facts.
- `expected_hallucination_type`: Expected hallucination error category for negative test vectors.

## Data Governance & Privacy

- **No PII**: All benchmark items use public domain scientific and historical reference facts.
