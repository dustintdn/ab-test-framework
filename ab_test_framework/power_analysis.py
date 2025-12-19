"""
Power Analysis Calculator

Helps determine required sample sizes for experiments and
evaluate the power of completed experiments.
"""

import numpy as np
from scipy import stats
from typing import Optional


class PowerAnalyzer:
    """
    Calculate statistical power and required sample sizes for A/B tests.
    
    Power is the probability of detecting an effect when it truly exists.
    Typically, we aim for 80% power or higher.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the power analyzer.
        
        Parameters
        ----------
        alpha : float
            Significance level (Type I error rate). Default is 0.05 (5%).
        """
        self.alpha = alpha
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
        ratio: float = 1.0
    ) -> dict:
        """
        Calculate required sample size per group for a proportion test.
        
        Parameters
        ----------
        baseline_rate : float
            Expected conversion rate in control group (between 0 and 1)
        minimum_detectable_effect : float
            Smallest effect size you want to detect (e.g., 0.02 for 2 percentage points)
        power : float
            Desired statistical power (typically 0.8 for 80%)
        ratio : float
            Ratio of treatment to control group size (1.0 means equal sizes)
        
        Returns
        -------
        dict
            Dictionary containing sample sizes and assumptions
        
        Example
        -------
        >>> analyzer = PowerAnalyzer()
        >>> result = analyzer.calculate_sample_size(
        ...     baseline_rate=0.10,
        ...     minimum_detectable_effect=0.02,
        ...     power=0.8
        ... )
        >>> print(f"Need {result['n_control']} per group")
        """
        # Get critical values from normal distribution
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)  # Two-tailed test
        z_beta = stats.norm.ppf(power)
        
        # Expected rate in treatment group
        treatment_rate = baseline_rate + minimum_detectable_effect
        
        # Pooled proportion for variance calculation
        pooled_rate = (baseline_rate + ratio * treatment_rate) / (1 + ratio)
        
        # Calculate required sample size for control group
        numerator = (z_alpha * np.sqrt(pooled_rate * (1 - pooled_rate) * (1 + 1/ratio)) +
                     z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + 
                                     treatment_rate * (1 - treatment_rate) / ratio))
        
        denominator = minimum_detectable_effect
        
        n_control = int(np.ceil((numerator / denominator) ** 2))
        n_treatment = int(np.ceil(n_control * ratio))
        
        return {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "total_sample_size": n_control + n_treatment,
            "assumptions": {
                "baseline_rate": baseline_rate,
                "minimum_detectable_effect": minimum_detectable_effect,
                "expected_treatment_rate": treatment_rate,
                "power": power,
                "alpha": self.alpha,
                "ratio": ratio
            }
        }
    
    def calculate_power(
        self,
        n_control: int,
        n_treatment: int,
        baseline_rate: float,
        treatment_rate: float
    ) -> dict:
        """
        Calculate the statistical power for a given sample size and effect.
        
        Use this to evaluate whether a completed or planned experiment
        has sufficient power to detect the observed/expected effect.
        
        Parameters
        ----------
        n_control : int
            Sample size in control group
        n_treatment : int
            Sample size in treatment group
        baseline_rate : float
            Conversion rate in control group
        treatment_rate : float
            Conversion rate in treatment group
        
        Returns
        -------
        dict
            Dictionary containing power calculation results
        
        Example
        -------
        >>> analyzer = PowerAnalyzer()
        >>> result = analyzer.calculate_power(
        ...     n_control=1000,
        ...     n_treatment=1000,
        ...     baseline_rate=0.10,
        ...     treatment_rate=0.12
        ... )
        >>> print(f"Power: {result['power']:.1%}")
        """
        effect_size = treatment_rate - baseline_rate
        
        # Standard error under alternative hypothesis
        se = np.sqrt(
            baseline_rate * (1 - baseline_rate) / n_control +
            treatment_rate * (1 - treatment_rate) / n_treatment
        )
        
        # Pooled proportion under null hypothesis
        pooled_rate = (baseline_rate * n_control + treatment_rate * n_treatment) / (n_control + n_treatment)
        se_null = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_control + 1/n_treatment))
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        
        # Non-centrality parameter
        z_beta = (abs(effect_size) - z_alpha * se_null) / se
        
        # Power
        power = stats.norm.cdf(z_beta)
        
        return {
            "power": power,
            "effect_size": effect_size,
            "effect_size_relative": effect_size / baseline_rate if baseline_rate > 0 else None,
            "sample_sizes": {
                "control": n_control,
                "treatment": n_treatment
            },
            "rates": {
                "control": baseline_rate,
                "treatment": treatment_rate
            },
            "interpretation": self._interpret_power(power)
        }
    
    def _interpret_power(self, power: float) -> str:
        """Provide interpretation of power value."""
        if power >= 0.9:
            return "Excellent power - very likely to detect true effects"
        elif power >= 0.8:
            return "Good power - acceptable for most applications"
        elif power >= 0.6:
            return "Moderate power - may miss true effects"
        else:
            return "Low power - high risk of missing true effects"
