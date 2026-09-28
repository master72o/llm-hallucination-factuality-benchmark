"""
Atomic Claim Extractor for Factuality & Evidence Verification.
"""

import re
from typing import List
from factuality_benchmark.schema import AtomicClaim, ClaimStatus, HallucinationType


class ClaimExtractor:
    """Extracts sentence-level atomic claims from LLM responses."""

    @staticmethod
    def extract_atomic_claims(response: str) -> List[AtomicClaim]:
        # Split into sentences using regex
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        claims: List[AtomicClaim] = []

        for idx, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue

            claims.append(
                AtomicClaim(
                    id=f"claim_{idx}",
                    text=sentence,
                    status=ClaimStatus.UNSUPPORTED,
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    explanation="Awaiting evidence verification."
                )
            )

        return claims
