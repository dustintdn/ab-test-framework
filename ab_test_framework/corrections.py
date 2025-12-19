"""
Multiple Testing Correction

Adjust p-values when testing multiple hypotheses to control
the family-wise error rate or false discovery rate.
"""

import numpy as np
from typing import List, Union


class MultipleTestingCorrection:
    """
    Apply corrections for multiple hypothesis testing.
    
    When running multiple tests, we need to adjust our significance
    threshold to avoid false positives (Type I errors).
    """
    
    def bonferroni(
        self,
        p_values: Union[List[float], np.ndarray],
        alpha: float = 0.05
    ) -> dict:
        """
        Apply Bonferroni correction (most conservative method).
        
        Bonferroni controls the Family-Wise Error Rate (FWER) - the probability
        of making at least one Type I error across all tests.
        
        Adjusted alpha = original alpha / number of tests
        
        Parameters
        ----------
        p_values : list or array
            Original p-values from multiple tests
        alpha : float
            Desired family-wise error rate (typically 0.05)
        
        Returns
        -------
        dict
            Corrected results with adjusted p-values and decisions
        
        Example
        -------
        >>> corrector = MultipleTestingCorrection()
        >>> p_values = [0.01, 0.04, 0.08, 0.12, 0.25]
        >>> result = corrector.bonferroni(p_values, alpha=0.05)
        >>> print(result['significant'])  # Which tests are still significant
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        # Adjusted alpha (more stringent)
        adjusted_alpha = alpha / n_tests
        
        # Adjusted p-values (more conservative)
        adjusted_p_values = np.minimum(p_values * n_tests, 1.0)
        
        # Significance decisions
        significant = adjusted_p_values < alpha
        
        return {
            "method": "Bonferroni",
            "original_alpha": alpha,
            "adjusted_alpha": adjusted_alpha,
            "n_tests": n_tests,
            "original_p_values": p_values.tolist(),
            "adjusted_p_values": adjusted_p_values.tolist(),
            "significant": significant.tolist(),
            "n_significant": int(np.sum(significant)),
            "interpretation": self._interpret_bonferroni(n_tests, adjusted_alpha)
        }
    
    def bonferroni_holm(
        self,
        p_values: Union[List[float], np.ndarray],
        alpha: float = 0.05
    ) -> dict:
        """
        Apply Bonferroni-Holm correction (less conservative than Bonferroni).
        
        A step-down procedure that provides more power than standard Bonferroni
        while still controlling FWER.
        
        Parameters
        ----------
        p_values : list or array
            Original p-values from multiple tests
        alpha : float
            Desired family-wise error rate
        
        Returns
        -------
        dict
            Corrected results with sequential decisions
        
        Example
        -------
        >>> corrector = MultipleTestingCorrection()
        >>> p_values = [0.01, 0.04, 0.08, 0.12, 0.25]
        >>> result = corrector.bonferroni_holm(p_values)
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Bonferroni-Holm sequential testing
        significant = np.zeros(n_tests, dtype=bool)
        adjusted_p_values = np.zeros(n_tests)
        
        for i, p in enumerate(sorted_p):
            # Adjusted alpha for this step
            step_alpha = alpha / (n_tests - i)
            
            # Adjusted p-value
            adjusted_p = p * (n_tests - i)
            adjusted_p_values[i] = min(adjusted_p, 1.0)
            
            # Stop if we fail to reject
            if p > step_alpha:
                break
            
            significant[i] = True
        
        # Reorder to match original order
        reordered_significant = np.zeros(n_tests, dtype=bool)
        reordered_adjusted_p = np.zeros(n_tests)
        
        for i, idx in enumerate(sorted_indices):
            reordered_significant[idx] = significant[i]
            reordered_adjusted_p[idx] = adjusted_p_values[i]
        
        return {
            "method": "Bonferroni-Holm",
            "original_alpha": alpha,
            "n_tests": n_tests,
            "original_p_values": p_values.tolist(),
            "adjusted_p_values": reordered_adjusted_p.tolist(),
            "significant": reordered_significant.tolist(),
            "n_significant": int(np.sum(reordered_significant)),
            "interpretation": "Step-down procedure, less conservative than standard Bonferroni"
        }
    
    def benjamini_hochberg(
        self,
        p_values: Union[List[float], np.ndarray],
        alpha: float = 0.05
    ) -> dict:
        """
        Apply Benjamini-Hochberg correction (controls False Discovery Rate).
        
        FDR controls the expected proportion of false positives among
        all rejections. Less conservative than Bonferroni methods,
        more appropriate when multiple discoveries are acceptable.
        
        Parameters
        ----------
        p_values : list or array
            Original p-values from multiple tests
        alpha : float
            Desired false discovery rate (typically 0.05 or 0.10)
        
        Returns
        -------
        dict
            Corrected results controlling FDR
        
        Example
        -------
        >>> corrector = MultipleTestingCorrection()
        >>> p_values = [0.01, 0.04, 0.08, 0.12, 0.25]
        >>> result = corrector.benjamini_hochberg(p_values, alpha=0.10)
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Benjamini-Hochberg procedure
        significant = np.zeros(n_tests, dtype=bool)
        adjusted_p_values = np.zeros(n_tests)
        
        # Work backwards to find largest i where p_i <= (i/m) * alpha
        threshold_found = False
        
        for i in range(n_tests - 1, -1, -1):
            rank = i + 1
            threshold = (rank / n_tests) * alpha
            adjusted_p = sorted_p[i] * n_tests / rank
            adjusted_p_values[i] = min(adjusted_p, 1.0)
            
            if sorted_p[i] <= threshold and not threshold_found:
                significant[:rank] = True
                threshold_found = True
        
        # Ensure monotonicity of adjusted p-values (from smallest to largest)
        for i in range(n_tests - 1, 0, -1):
            if adjusted_p_values[i] < adjusted_p_values[i-1]:
                adjusted_p_values[i-1] = adjusted_p_values[i]
        
        # Reorder to match original order
        reordered_significant = np.zeros(n_tests, dtype=bool)
        reordered_adjusted_p = np.zeros(n_tests)
        
        for i, idx in enumerate(sorted_indices):
            reordered_significant[idx] = significant[i]
            reordered_adjusted_p[idx] = adjusted_p_values[i]
        
        return {
            "method": "Benjamini-Hochberg",
            "original_alpha": alpha,
            "n_tests": n_tests,
            "original_p_values": p_values.tolist(),
            "adjusted_p_values": reordered_adjusted_p.tolist(),
            "significant": reordered_significant.tolist(),
            "n_significant": int(np.sum(reordered_significant)),
            "interpretation": self._interpret_bh(alpha)
        }
    
    def compare_methods(
        self,
        p_values: Union[List[float], np.ndarray],
        alpha: float = 0.05
    ) -> dict:
        """
        Compare all correction methods side-by-side.
        
        Useful for understanding the trade-offs between methods.
        
        Parameters
        ----------
        p_values : list or array
            Original p-values from multiple tests
        alpha : float
            Desired error rate
        
        Returns
        -------
        dict
            Comparison of all methods
        
        Example
        -------
        >>> corrector = MultipleTestingCorrection()
        >>> p_values = [0.01, 0.04, 0.08, 0.12, 0.25]
        >>> comparison = corrector.compare_methods(p_values)
        >>> print(comparison['summary'])
        """
        # Apply all methods
        bonf = self.bonferroni(p_values, alpha)
        holm = self.bonferroni_holm(p_values, alpha)
        bh = self.benjamini_hochberg(p_values, alpha)
        
        # Create summary
        summary = {
            "n_tests": len(p_values),
            "original_alpha": alpha,
            "uncorrected_significant": int(np.sum(np.array(p_values) < alpha)),
            "bonferroni_significant": bonf["n_significant"],
            "bonferroni_holm_significant": holm["n_significant"],
            "benjamini_hochberg_significant": bh["n_significant"]
        }
        
        return {
            "summary": summary,
            "bonferroni": bonf,
            "bonferroni_holm": holm,
            "benjamini_hochberg": bh,
            "recommendation": self._recommend_method(summary)
        }
    
    def _interpret_bonferroni(self, n_tests: int, adjusted_alpha: float) -> str:
        """Provide interpretation of Bonferroni correction."""
        return (f"Very conservative method. With {n_tests} tests, "
                f"individual test alpha is reduced to {adjusted_alpha:.4f}. "
                f"Use when avoiding any false positives is critical.")
    
    def _interpret_bh(self, alpha: float) -> str:
        """Provide interpretation of Benjamini-Hochberg correction."""
        return (f"Controls False Discovery Rate at {alpha}. "
                f"Accepts that {alpha*100:.0f}% of significant results may be false positives. "
                f"More powerful than Bonferroni, good for exploratory analysis.")
    
    def _recommend_method(self, summary: dict) -> str:
        """Recommend which correction method to use."""
        n_tests = summary["n_tests"]
        
        if n_tests <= 3:
            return "With few tests, Bonferroni is acceptable and simple"
        elif n_tests <= 10:
            return "Bonferroni-Holm offers good balance - less conservative than Bonferroni"
        else:
            return ("Benjamini-Hochberg recommended for many tests - "
                   "controls FDR while maintaining good power")
