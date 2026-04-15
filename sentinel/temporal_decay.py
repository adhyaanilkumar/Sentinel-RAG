"""Temporal decay mechanism for time-sensitive military intelligence."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from core.data_models import DocumentChunk


class TemporalDecayEngine:
    """Apply temporal decay weights to graph nodes based on document timestamps."""

    def __init__(
        self,
        decay_function: str = "exponential",
        half_life_hours: float = 72,
        stale_threshold_hours: float = 168,
        flag_stale: bool = True,
        reference_time: Optional[datetime] = None,
    ):
        self.decay_function = decay_function
        self.half_life_hours = half_life_hours
        self.stale_threshold_hours = stale_threshold_hours
        self.flag_stale = flag_stale
        self.reference_time = reference_time or datetime.now(timezone.utc)

    def compute_weight(self, timestamp: Optional[datetime]) -> float:
        """Compute temporal weight for a single timestamp. Returns 1.0 if no timestamp."""
        if timestamp is None:
            return 1.0

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_hours = (self.reference_time - timestamp).total_seconds() / 3600.0
        if age_hours < 0:
            return 1.0

        if self.decay_function == "exponential":
            return math.exp(-math.log(2) * age_hours / self.half_life_hours)
        elif self.decay_function == "linear":
            return max(0.0, 1.0 - age_hours / (self.stale_threshold_hours * 2))
        elif self.decay_function == "step":
            return 0.3 if age_hours > self.stale_threshold_hours else 1.0
        return 1.0

    def is_stale(self, timestamp: Optional[datetime]) -> bool:
        if timestamp is None:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_hours = (self.reference_time - timestamp).total_seconds() / 3600.0
        return age_hours > self.stale_threshold_hours

    def compute_weights(self, knowledge_graph) -> dict[str, float]:
        """Compute temporal weights for all nodes in the knowledge graph."""
        weights = {}
        for nid, node in knowledge_graph.nodes.items():
            weights[nid] = self.compute_weight(node.chunk.timestamp)
        return weights

    def flag_stale_chunks(self, chunks: list[DocumentChunk]) -> str:
        """Generate stale-info warnings for retrieved chunks."""
        if not self.flag_stale:
            return ""

        warnings = []
        for chunk in chunks:
            if self.is_stale(chunk.timestamp):
                age = None
                if chunk.timestamp:
                    if chunk.timestamp.tzinfo is None:
                        ts = chunk.timestamp.replace(tzinfo=timezone.utc)
                    else:
                        ts = chunk.timestamp
                    age = (self.reference_time - ts).total_seconds() / 3600.0
                age_str = f" ({age:.0f}h old)" if age else ""
                warnings.append(
                    f"WARNING: {chunk.source_document} {chunk.section_id} may contain "
                    f"stale information{age_str}. Verify against more recent sources."
                )
        return "\n".join(warnings)
