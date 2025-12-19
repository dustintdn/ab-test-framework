"""
Result Visualization

Create clear, publication-ready visualizations of A/B test results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


class ResultVisualizer:
    """
    Create visualizations for A/B test results.
    
    All plots are designed to be clear, publication-ready,
    and easy to interpret.
    """
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        style : str
            Seaborn style (default: "whitegrid")
        """
        sns.set_style(style)
        self.colors = {
            "control": "#2E86AB",
            "treatment": "#A23B72",
            "neutral": "#6C757D"
        }
    
    def plot_conversion_rates(
        self,
        rate_control: float,
        rate_treatment: float,
        ci_control: Tuple[float, float],
        ci_treatment: Tuple[float, float],
        n_control: int,
        n_treatment: int,
        title: str = "Conversion Rate Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot conversion rates with confidence intervals.
        
        Parameters
        ----------
        rate_control : float
            Control group conversion rate
        rate_treatment : float
            Treatment group conversion rate
        ci_control : tuple
            95% CI for control rate (lower, upper)
        ci_treatment : tuple
            95% CI for treatment rate (lower, upper)
        n_control : int
            Sample size of control group
        n_treatment : int
            Sample size of treatment group
        title : str
            Plot title
        save_path : str, optional
            If provided, save plot to this path
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        
        Example
        -------
        >>> viz = ResultVisualizer()
        >>> fig = viz.plot_conversion_rates(
        ...     rate_control=0.10,
        ...     rate_treatment=0.13,
        ...     ci_control=(0.08, 0.12),
        ...     ci_treatment=(0.11, 0.15),
        ...     n_control=1000,
        ...     n_treatment=1000
        ... )
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = ["Control", "Treatment"]
        rates = [rate_control, rate_treatment]
        cis = [ci_control, ci_treatment]
        ns = [n_control, n_treatment]
        colors = [self.colors["control"], self.colors["treatment"]]
        
        # Create bar plot
        x_pos = np.arange(len(groups))
        bars = ax.bar(x_pos, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add confidence intervals
        for i, (rate, ci) in enumerate(zip(rates, cis)):
            error = [[rate - ci[0]], [ci[1] - rate]]
            ax.errorbar(i, rate, yerr=error, fmt='none', color='black', 
                       capsize=10, capthick=2, linewidth=2)
        
        # Add value labels on bars
        for i, (bar, rate, n) in enumerate(zip(bars, rates, ns)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.2%}\n(n={n:,})',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Formatting
        ax.set_ylabel('Conversion Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_effect_size(
        self,
        effect_size: float,
        ci_lower: float,
        ci_upper: float,
        metric_name: str = "Effect Size",
        title: str = "Effect Size with 95% Confidence Interval",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot effect size with confidence interval.
        
        Parameters
        ----------
        effect_size : float
            Observed effect size
        ci_lower : float
            Lower bound of confidence interval
        ci_upper : float
            Upper bound of confidence interval
        metric_name : str
            Name of the metric
        title : str
            Plot title
        save_path : str, optional
            If provided, save plot to this path
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot confidence interval
        ax.plot([ci_lower, ci_upper], [0, 0], 'o-', linewidth=3, 
               markersize=8, color=self.colors["treatment"], label='95% CI')
        
        # Plot point estimate
        ax.plot(effect_size, 0, 'D', markersize=15, color=self.colors["control"], 
               label='Point Estimate', zorder=5)
        
        # Reference line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='No Effect')
        
        # Annotations
        ax.text(effect_size, 0.05, f'{effect_size:.4f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Formatting
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distributions(
        self,
        values_control: np.ndarray,
        values_treatment: np.ndarray,
        title: str = "Distribution Comparison",
        metric_name: str = "Metric Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot overlapping distributions of control and treatment groups.
        
        Parameters
        ----------
        values_control : array-like
            Values from control group
        values_treatment : array-like
            Values from treatment group
        title : str
            Plot title
        metric_name : str
            Name of the metric being plotted
        save_path : str, optional
            If provided, save plot to this path
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot distributions
        ax.hist(values_control, bins=30, alpha=0.6, color=self.colors["control"], 
               label=f'Control (n={len(values_control):,})', density=True, edgecolor='black')
        ax.hist(values_treatment, bins=30, alpha=0.6, color=self.colors["treatment"], 
               label=f'Treatment (n={len(values_treatment):,})', density=True, edgecolor='black')
        
        # Add mean lines
        mean_control = np.mean(values_control)
        mean_treatment = np.mean(values_treatment)
        
        ax.axvline(mean_control, color=self.colors["control"], linestyle='--', 
                  linewidth=2, label=f'Control Mean: {mean_control:.2f}')
        ax.axvline(mean_treatment, color=self.colors["treatment"], linestyle='--', 
                  linewidth=2, label=f'Treatment Mean: {mean_treatment:.2f}')
        
        # Formatting
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bootstrap_distribution(
        self,
        bootstrap_diffs: np.ndarray,
        observed_diff: float,
        ci_lower: float,
        ci_upper: float,
        title: str = "Bootstrap Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bootstrap sampling distribution.
        
        Parameters
        ----------
        bootstrap_diffs : array
            Bootstrap differences
        observed_diff : float
            Observed difference
        ci_lower : float
            Lower CI bound
        ci_upper : float
            Upper CI bound
        title : str
            Plot title
        save_path : str, optional
            If provided, save plot to this path
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram
        ax.hist(bootstrap_diffs, bins=50, alpha=0.7, color=self.colors["neutral"], 
               edgecolor='black', density=True)
        
        # Mark observed difference
        ax.axvline(observed_diff, color=self.colors["treatment"], linestyle='-', 
                  linewidth=3, label=f'Observed: {observed_diff:.4f}')
        
        # Mark confidence interval
        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, 
                  label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2)
        
        # Shade CI region
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red')
        
        # Zero line
        ax.axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Difference', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_tests(
        self,
        test_names: List[str],
        p_values: List[float],
        adjusted_p_values: List[float],
        alpha: float = 0.05,
        title: str = "Multiple Testing Correction Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize multiple testing correction results.
        
        Parameters
        ----------
        test_names : list
            Names of tests
        p_values : list
            Original p-values
        adjusted_p_values : list
            Corrected p-values
        alpha : float
            Significance threshold
        title : str
            Plot title
        save_path : str, optional
            If provided, save plot to this path
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(test_names))
        width = 0.35
        
        # Plot original and adjusted p-values
        bars1 = ax.bar(x - width/2, p_values, width, label='Original p-value',
                      color=self.colors["control"], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, adjusted_p_values, width, label='Adjusted p-value',
                      color=self.colors["treatment"], alpha=0.7, edgecolor='black')
        
        # Significance threshold line
        ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2, 
                  label=f'α = {alpha}', zorder=0)
        
        # Formatting
        ax.set_ylabel('P-value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
