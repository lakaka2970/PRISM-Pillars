"""
Radar evidence modules for PRISM-Pillars-RF.

Module B: STER - Self-Supervised Temporal Evidence Reliability
Module C: DAUT - Doppler-Aware Uncertainty Tube
Module D: RAPR - Reliability-Aware Probabilistic Routing

These three modules form the paper's core contributions:
    1. Anisotropic probabilistic history evidence
    2. Self-supervised temporal reliability
    3. Reliability-weighted probabilistic pillar routing
"""

from .radar_point_embedding import RadarPointEmbedding
from .temporal_reliability import TemporalReliabilityEstimator
from .temporal_support_builder import TemporalSupportBuilder
from .doppler_uncertainty_tube import DopplerUncertaintyTube
from .probabilistic_pillar_router import ProbabilisticPillarRouter

__all__ = {
    'RadarPointEmbedding': RadarPointEmbedding,
    'TemporalReliabilityEstimator': TemporalReliabilityEstimator,
    'TemporalSupportBuilder': TemporalSupportBuilder,
    'DopplerUncertaintyTube': DopplerUncertaintyTube,
    'ProbabilisticPillarRouter': ProbabilisticPillarRouter,
}
