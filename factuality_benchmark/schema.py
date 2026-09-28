"""
Data Schemas for LLM Hallucination & Factuality Benchmark.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from factuality_benchmark.taxonomy import HallucinationType, ClaimStatus


@dataclass
class AtomicClaim:
    id: str
    text: str
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    hallucination_type: HallucinationType = HallucinationType.UNSUPPORTED_CLAIM
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status.value if isinstance(self.status, ClaimStatus) else self.status,
            "hallucination_type": self.hallucination_type.value if isinstance(self.hallucination_type, HallucinationType) else self.hallucination_type,
            "explanation": self.explanation,
        }


@dataclass
class FactualityItem:
    id: str
    prompt: str
    response: str
    evidence_text: str
    expected_hallucination_type: Optional[HallucinationType] = None
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactualityItem":
        exp_h = HallucinationType(data["expected_hallucination_type"]) if data.get("expected_hallucination_type") else None
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response=data["response"],
            evidence_text=data["evidence_text"],
            expected_hallucination_type=exp_h,
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "evidence_text": self.evidence_text,
            "expected_hallucination_type": self.expected_hallucination_type.value if self.expected_hallucination_type else None,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class FactualityResult:
    item_id: str
    prompt: str
    response: str
    evidence_text: str
    claims: List[AtomicClaim]
    precision: float
    recall: float
    f1_score: float
    primary_hallucination_type: HallucinationType
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "response": self.response,
            "evidence_text": self.evidence_text,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "passed": self.passed,
            "primary_hallucination_type": self.primary_hallucination_type.value if isinstance(self.primary_hallucination_type, HallucinationType) else self.primary_hallucination_type,
            "claims": [c.to_dict() for c in self.claims],
        }
