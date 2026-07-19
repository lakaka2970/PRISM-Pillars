"""
Temporal fusion modules for PRISM-Pillars-RF.

Module E: CRLF - Causal Reliability-Aware Local Fusion

Implements causal local pillar attention where current pillars query
historical probability evidence within a local radius, using
Mahalanobis-aware attention scoring and gated residual fusion.
"""

from .local_candidate_retriever import LocalCandidateRetriever
from .mahalanobis_bias import MahalanobisBias
from .causal_local_pillar_fusion import CausalLocalPillarFusion

__all__ = {
    'LocalCandidateRetriever': LocalCandidateRetriever,
    'MahalanobisBias': MahalanobisBias,
    'CausalLocalPillarFusion': CausalLocalPillarFusion,
}
