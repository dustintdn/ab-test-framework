"""
Effect Size Estimation

Calculate effect sizes with confidence intervals to understand
the practical significance of A/B test results.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


class EffectSizeCalculator:
    """
    Calculate effect sizes and confidence intervals.
    
    Effect sizes help us understand the practical significance
    of differences, not just statistical significance.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize effect size calculator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals (default 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def cohens_d(
        self,
        values_control: np.ndarray,
        values_treatment: np.ndarray
    ) -> dict:
        """
        Calculate Cohen's d effect size for continuous metrics.
        
        Cohen's d measures the standardized difference between two means.
        Rules of thumb: 0.2 = small, 0.5 = medium, 0.8 = large effect.
        
        Parameters
        ----------
        values_control : array-like
            Values from control group
        values_treatment : array-like
            Values from treatment group
        
        Returns
        -------
        dict
            Cohen's d with confidence interval and interpretation
        
        Example
        -------
        >>> calc = EffectSizeCalculator()
        >>> control = np.random.normal(100, 15, 1000)
        >>> treatment = np.random.normal(105, 15, 1000)
        >>> result = calc.cohens_d(control, treatment)
        >>> print(f"Cohen's d: {result['cohens_d']:.3f}")
        """
        values_control = np.asarray(values_control)
        values_treatment = np.asarray(values_treatment)
        
        n1, n2 = len(values_control), len(values_treatment)
        mean1, mean2 = np.mean(values_control), np.mean(values_treatment)
        var1, var2 = np.var(values_control, ddof=1), np.var(values_treatment, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_std
        
        # Confidence interval for Cohen's d
        # Using non-central t distribution
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        ci_lower = d - stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2) * se
        ci_upper = d + stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2) * se
        
        return {
            "cohens_d": d,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": self.confidence_level,
            "interpretation": self._interpret_cohens_d(d),
            "means": {
                "control": mean1,
                "treatment": mean2,
                "difference": mean2 - mean1
            }
        }
    
    def absolute_difference_ci(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int
    ) -> dict:
        """
        Calculate absolute difference in proportions with confidence interval.
        
        For conversion rates, click-through rates, etc.
        
        Parameters
        ----------
        conversions_control : int
            Number of conversions in control
        n_control : int
            Total size of control group
        conversions_treatment : int
            Number of conversions in treatment
        n_treatment : int
            Total size of treatment group
        
        Returns
        -------
        dict
            Absolute difference with confidence interval
        
        Example
        -------
        >>> calc = EffectSizeCalculator()
        >>> result = calc.absolute_difference_ci(
        ...     conversions_control=100,
        ...     n_control=1000,
        ...     conversions_treatment=130,
        ...     n_treatment=1000
        ... )
        >>> print(f"Absolute lift: {result['absolute_difference']:.1%}")
        """
        p1 = conversions_control / n_control
        p2 = conversions_treatment / n_treatment
        
        diff = p2 - p1
        
        # Standard error for difference
        se = np.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_treatment)
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = diff - z_critical * se
        ci_upper = diff + z_critical * se
        
        return {
            "absolute_difference": diff,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": self.confidence_level,
            "rates": {
                "control": p1,
                "treatment": p2
            },
            "interpretation": self._interpret_proportion_diff(diff, p1)
        }
    
    def relative_lift_ci(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int,
        method: str = "delta"
    ) -> dict:
        """
        Calculate relative lift (percentage change) with confidence interval.
        
        Relative lift = (Treatment Rate - Control Rate) / Control Rate
        
        Parameters
        ----------
        conversions_control : int
            Number of conversions in control
        n_control : int
            Total size of control group
        conversions_treatment : int
            Number of conversions in treatment
        n_treatment : int
            Total size of treatment group
        method : str
            Method for CI calculation: "delta" (default) or "log"
        
        Returns
        -------
        dict
            Relative lift with confidence interval
        
        Example
        -------
        >>> calc = EffectSizeCalculator()
        >>> result = calc.relative_lift_ci(
        ...     conversions_control=100,
        ...     n_control=1000,
        ...     conversions_treatment=130,
        ...     n_treatment=1000
        ... )
        >>> print(f"Relative lift: {result['relative_lift']:.1%}")
        """
        p1 = conversions_control / n_control
        p2 = conversions_treatment / n_treatment
        
        if p1 == 0:
            return {
                "relative_lift": None,
                "confidence_interval": (None, None),
                "error": "Control rate is zero - cannot calculate relative lift"
            }
        
        relative_lift = (p2 - p1) / p1
        
        if method == "delta":
            # Delta method for confidence interval
            se_p1 = np.sqrt(p1 * (1 - p1) / n_control)
            se_p2 = np.sqrt(p2 * (1 - p2) / n_treatment)
            
            # Taylor approximation
            se_lift = np.sqrt((se_p2 / p1)**2 + ((p2 * se_p1) / p1**2)**2)
            
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
            ci_lower = relative_lift - z_critical * se_lift
            ci_upper = relative_lift + z_critical * se_lift
            
        else:  # log method
            # Log transformation for better properties
            log_rr = np.log(p2 / p1)
            se_log = np.sqrt((1 - p1) / (n_control * p1) + (1 - p2) / (n_treatment * p2))
            
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
            ci_lower = np.exp(log_rr - z_critical * se_log) - 1
            ci_upper = np.exp(log_rr + z_critical * se_log) - 1
        
        return {
            "relative_lift": relative_lift,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": self.confidence_level,
            "rates": {
                "control": p1,
                "treatment": p2,
                "absolute_difference": p2 - p1
            },
            "interpretation": self._interpret_relative_lift(relative_lift)
        }
    
    def bootstrap_ci(
        self,
        values_control: np.ndarray,
        values_treatment: np.ndarray,
        statistic: str = "mean",
        n_bootstrap: int = 10000,
        random_seed: Optional[int] = None
    ) -> dict:
        """
        Bootstrap confidence interval for any statistic.
        
        Useful for metrics where traditional CI formulas don't apply.
        
        Parameters
        ----------
        values_control : array-like
            Values from control group
        values_treatment : array-like
            Values from treatment group
        statistic : str
            Statistic to estimate: "mean", "median", or "proportion"
        n_bootstrap : int
            Number of bootstrap samples
        random_seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        dict
            Bootstrap confidence interval
        
        Example
        -------
        >>> calc = EffectSizeCalculator()
        >>> control = np.random.exponential(100, 1000)
        >>> treatment = np.random.exponential(110, 1000)
        >>> result = calc.bootstrap_ci(control, treatment, statistic="median")
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        values_control = np.asarray(values_control)
        values_treatment = np.asarray(values_treatment)
        
        # Choose statistic function
        if statistic == "mean":
            stat_func = np.mean
        elif statistic == "median":
            stat_func = np.median
        elif statistic == "proportion":
            stat_func = lambda x: np.sum(x) / len(x)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Bootstrap resampling
        bootstrap_diffs = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            boot_control = np.random.choice(values_control, size=len(values_control), replace=True)
            boot_treatment = np.random.choice(values_treatment, size=len(values_treatment), replace=True)
            
            bootstrap_diffs[i] = stat_func(boot_treatment) - stat_func(boot_control)
        
        # Percentile method for CI
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))
        
        observed_diff = stat_func(values_treatment) - stat_func(values_control)
        
        return {
            "effect_size": observed_diff,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": self.confidence_level,
            "method": f"Bootstrap ({n_bootstrap} resamples)",
            "values": {
                "control": stat_func(values_control),
                "treatment": stat_func(values_treatment)
            }
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        d_abs = abs(d)
        if d_abs < 0.2:
            magnitude = "negligible"
        elif d_abs < 0.5:
            magnitude = "small"
        elif d_abs < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "positive" if d > 0 else "negative"
        return f"{magnitude.capitalize()} {direction} effect (|d| = {d_abs:.2f})"
    
    def _interpret_proportion_diff(self, diff: float, baseline: float) -> str:
        """Interpret absolute difference in proportions."""
        percentage_points = diff * 100
        relative = (diff / baseline * 100) if baseline > 0 else 0
        
        return f"{percentage_points:+.2f} percentage points ({relative:+.1f}% relative change)"
    
    def _interpret_relative_lift(self, lift: float) -> str:
        """Interpret relative lift magnitude."""
        lift_pct = lift * 100
        
        if abs(lift_pct) < 1:
            magnitude = "negligible"
        elif abs(lift_pct) < 5:
            magnitude = "small"
        elif abs(lift_pct) < 15:
            magnitude = "moderate"
        else:
            magnitude = "substantial"
        
        direction = "increase" if lift > 0 else "decrease"
        return f"{magnitude.capitalize()} {direction} of {abs(lift_pct):.1f}%"
