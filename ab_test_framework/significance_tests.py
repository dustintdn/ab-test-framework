"""
Significance Testing

Includes both parametric and bootstrap-based significance tests
for A/B test analysis.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


class SignificanceTest:
    """
    Perform significance tests for A/B experiments.
    
    Supports both traditional parametric tests and bootstrap-based
    non-parametric tests.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize significance tester.
        
        Parameters
        ----------
        alpha : float
            Significance level (typically 0.05 for 5%)
        """
        self.alpha = alpha
    
    def proportions_test(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int
    ) -> dict:
        """
        Test for difference in proportions (e.g., conversion rates).
        
        Uses a two-proportion z-test.
        
        Parameters
        ----------
        conversions_control : int
            Number of conversions in control group
        n_control : int
            Total size of control group
        conversions_treatment : int
            Number of conversions in treatment group
        n_treatment : int
            Total size of treatment group
        
        Returns
        -------
        dict
            Test results including p-value and conclusion
        
        Example
        -------
        >>> tester = SignificanceTest()
        >>> result = tester.proportions_test(
        ...     conversions_control=100,
        ...     n_control=1000,
        ...     conversions_treatment=130,
        ...     n_treatment=1000
        ... )
        >>> print(f"P-value: {result['p_value']:.4f}")
        """
        rate_control = conversions_control / n_control
        rate_treatment = conversions_treatment / n_treatment
        
        # Pooled proportion
        pooled_rate = (conversions_control + conversions_treatment) / (n_control + n_treatment)
        
        # Standard error under null hypothesis
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_control + 1/n_treatment))
        
        # Z-statistic
        z_stat = (rate_treatment - rate_control) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "test_type": "Two-proportion z-test",
            "statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "rates": {
                "control": rate_control,
                "treatment": rate_treatment,
                "difference": rate_treatment - rate_control,
                "relative_lift": (rate_treatment - rate_control) / rate_control if rate_control > 0 else None
            },
            "interpretation": self._interpret_result(p_value, rate_treatment - rate_control)
        }
    
    def means_test(
        self,
        values_control: np.ndarray,
        values_treatment: np.ndarray,
        equal_var: bool = False
    ) -> dict:
        """
        Test for difference in means (e.g., revenue, time on site).
        
        Uses Welch's t-test by default (does not assume equal variances).
        
        Parameters
        ----------
        values_control : array-like
            Continuous values from control group
        values_treatment : array-like
            Continuous values from treatment group
        equal_var : bool
            If True, assumes equal variances (Student's t-test)
        
        Returns
        -------
        dict
            Test results including p-value and conclusion
        
        Example
        -------
        >>> tester = SignificanceTest()
        >>> control = np.random.normal(100, 15, 1000)
        >>> treatment = np.random.normal(105, 15, 1000)
        >>> result = tester.means_test(control, treatment)
        """
        values_control = np.asarray(values_control)
        values_treatment = np.asarray(values_treatment)
        
        mean_control = np.mean(values_control)
        mean_treatment = np.mean(values_treatment)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            values_treatment,
            values_control,
            equal_var=equal_var
        )
        
        test_name = "Student's t-test" if equal_var else "Welch's t-test"
        
        return {
            "test_type": test_name,
            "statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "means": {
                "control": mean_control,
                "treatment": mean_treatment,
                "difference": mean_treatment - mean_control,
                "relative_lift": (mean_treatment - mean_control) / mean_control if mean_control != 0 else None
            },
            "interpretation": self._interpret_result(p_value, mean_treatment - mean_control)
        }
    
    def bootstrap_test(
        self,
        values_control: np.ndarray,
        values_treatment: np.ndarray,
        n_bootstrap: int = 10000,
        statistic: str = "mean",
        random_seed: Optional[int] = None
    ) -> dict:
        """
        Bootstrap-based permutation test for any statistic.
        
        Non-parametric test that doesn't assume any distribution.
        Useful for metrics with skewed distributions or outliers.
        
        Parameters
        ----------
        values_control : array-like
            Values from control group
        values_treatment : array-like
            Values from treatment group
        n_bootstrap : int
            Number of bootstrap resamples (default 10,000)
        statistic : str
            Statistic to test: "mean", "median", or "proportion"
        random_seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        dict
            Bootstrap test results with empirical p-value
        
        Example
        -------
        >>> tester = SignificanceTest()
        >>> control = np.random.exponential(100, 1000)
        >>> treatment = np.random.exponential(110, 1000)
        >>> result = tester.bootstrap_test(control, treatment, statistic="median")
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
        
        # Observed difference
        observed_diff = stat_func(values_treatment) - stat_func(values_control)
        
        # Combine all data for permutation
        combined = np.concatenate([values_control, values_treatment])
        n_control = len(values_control)
        n_treatment = len(values_treatment)
        
        # Bootstrap resampling under null hypothesis (no difference)
        bootstrap_diffs = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Randomly shuffle and split
            shuffled = np.random.permutation(combined)
            boot_control = shuffled[:n_control]
            boot_treatment = shuffled[n_control:n_control + n_treatment]
            
            # Calculate difference
            bootstrap_diffs[i] = stat_func(boot_treatment) - stat_func(boot_control)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return {
            "test_type": f"Bootstrap permutation test ({statistic})",
            "observed_difference": observed_diff,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "n_bootstrap": n_bootstrap,
            "bootstrap_distribution": {
                "mean": np.mean(bootstrap_diffs),
                "std": np.std(bootstrap_diffs),
                "percentile_2_5": np.percentile(bootstrap_diffs, 2.5),
                "percentile_97_5": np.percentile(bootstrap_diffs, 97.5)
            },
            "values": {
                "control": stat_func(values_control),
                "treatment": stat_func(values_treatment)
            },
            "interpretation": self._interpret_result(p_value, observed_diff)
        }
    
    def _interpret_result(self, p_value: float, effect: float) -> str:
        """Provide plain-language interpretation of test result."""
        direction = "positive" if effect > 0 else "negative"
        
        if p_value < 0.001:
            return f"Very strong evidence of a {direction} effect"
        elif p_value < 0.01:
            return f"Strong evidence of a {direction} effect"
        elif p_value < self.alpha:
            return f"Evidence of a {direction} effect"
        elif p_value < 0.10:
            return f"Weak evidence of a {direction} effect (not significant at α={self.alpha})"
        else:
            return "No significant evidence of an effect"
