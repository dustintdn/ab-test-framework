"""
A/B Test Analysis Framework

A simple, readable framework for analyzing A/B tests and experiments.
Designed for Data Scientists who need clear, interpretable results.
"""

from .power_analysis import PowerAnalyzer
from .significance_tests import SignificanceTest
from .effect_size import EffectSizeCalculator
from .corrections import MultipleTestingCorrection
from .visualizations import ResultVisualizer

__version__ = "0.1.0"
__all__ = [
    "PowerAnalyzer",
    "SignificanceTest",
    "EffectSizeCalculator",
    "MultipleTestingCorrection",
    "ResultVisualizer",
]
