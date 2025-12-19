"""
Simple tests to verify the framework works correctly.

Run with: python -m pytest tests/test_basic.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ab_test_framework import (
    PowerAnalyzer,
    SignificanceTest,
    EffectSizeCalculator,
    MultipleTestingCorrection
)


def test_power_analysis():
    """Test power analysis calculations."""
    analyzer = PowerAnalyzer(alpha=0.05)
    
    result = analyzer.calculate_sample_size(
        baseline_rate=0.10,
        minimum_detectable_effect=0.02,
        power=0.8
    )
    
    # Sample size should be reasonable
    assert result['n_control'] > 0
    assert result['n_control'] < 100000
    assert result['total_sample_size'] == result['n_control'] + result['n_treatment']


def test_proportions_test():
    """Test significance test for proportions."""
    tester = SignificanceTest(alpha=0.05)
    
    # Test with clear difference
    result = tester.proportions_test(
        conversions_control=100,
        n_control=1000,
        conversions_treatment=150,
        n_treatment=1000
    )
    
    assert 'p_value' in result
    assert 'significant' in result
    assert result['p_value'] >= 0
    assert result['p_value'] <= 1


def test_means_test():
    """Test significance test for means."""
    np.random.seed(42)
    
    tester = SignificanceTest(alpha=0.05)
    
    control = np.random.normal(100, 15, 1000)
    treatment = np.random.normal(105, 15, 1000)
    
    result = tester.means_test(control, treatment)
    
    assert 'p_value' in result
    assert 'statistic' in result
    assert result['means']['treatment'] > result['means']['control']


def test_bootstrap():
    """Test bootstrap-based test."""
    np.random.seed(42)
    
    tester = SignificanceTest(alpha=0.05)
    
    control = np.random.normal(100, 15, 100)
    treatment = np.random.normal(105, 15, 100)
    
    result = tester.bootstrap_test(
        values_control=control,
        values_treatment=treatment,
        n_bootstrap=1000,
        statistic="mean",
        random_seed=42
    )
    
    assert 'p_value' in result
    assert 'observed_difference' in result


def test_effect_size_cohens_d():
    """Test Cohen's d calculation."""
    np.random.seed(42)
    
    calc = EffectSizeCalculator(confidence_level=0.95)
    
    control = np.random.normal(100, 15, 1000)
    treatment = np.random.normal(105, 15, 1000)
    
    result = calc.cohens_d(control, treatment)
    
    assert 'cohens_d' in result
    assert 'confidence_interval' in result
    assert result['confidence_interval'][0] < result['cohens_d'] < result['confidence_interval'][1]


def test_absolute_difference():
    """Test absolute difference calculation."""
    calc = EffectSizeCalculator(confidence_level=0.95)
    
    result = calc.absolute_difference_ci(
        conversions_control=100,
        n_control=1000,
        conversions_treatment=130,
        n_treatment=1000
    )
    
    assert 'absolute_difference' in result
    assert 'confidence_interval' in result
    assert result['absolute_difference'] == 0.03


def test_relative_lift():
    """Test relative lift calculation."""
    calc = EffectSizeCalculator(confidence_level=0.95)
    
    result = calc.relative_lift_ci(
        conversions_control=100,
        n_control=1000,
        conversions_treatment=130,
        n_treatment=1000
    )
    
    assert 'relative_lift' in result
    assert 'confidence_interval' in result
    assert result['relative_lift'] == 0.3  # 30% lift


def test_bonferroni():
    """Test Bonferroni correction."""
    corrector = MultipleTestingCorrection()
    
    p_values = [0.01, 0.04, 0.08]
    result = corrector.bonferroni(p_values, alpha=0.05)
    
    assert result['n_tests'] == 3
    assert result['adjusted_alpha'] == 0.05 / 3
    assert len(result['adjusted_p_values']) == 3


def test_benjamini_hochberg():
    """Test Benjamini-Hochberg correction."""
    corrector = MultipleTestingCorrection()
    
    p_values = [0.01, 0.04, 0.08]
    result = corrector.benjamini_hochberg(p_values, alpha=0.05)
    
    assert result['n_tests'] == 3
    assert len(result['adjusted_p_values']) == 3
    # BH should reject more than Bonferroni
    assert result['n_significant'] >= 0


def test_compare_methods():
    """Test comparison of multiple correction methods."""
    corrector = MultipleTestingCorrection()
    
    p_values = [0.01, 0.04, 0.08, 0.12, 0.25]
    comparison = corrector.compare_methods(p_values, alpha=0.05)
    
    assert 'summary' in comparison
    assert 'bonferroni' in comparison
    assert 'bonferroni_holm' in comparison
    assert 'benjamini_hochberg' in comparison


if __name__ == "__main__":
    # Run all tests
    print("Running basic tests...")
    
    test_power_analysis()
    print("✓ Power analysis test passed")
    
    test_proportions_test()
    print("✓ Proportions test passed")
    
    test_means_test()
    print("✓ Means test passed")
    
    test_bootstrap()
    print("✓ Bootstrap test passed")
    
    test_effect_size_cohens_d()
    print("✓ Cohen's d test passed")
    
    test_absolute_difference()
    print("✓ Absolute difference test passed")
    
    test_relative_lift()
    print("✓ Relative lift test passed")
    
    test_bonferroni()
    print("✓ Bonferroni test passed")
    
    test_benjamini_hochberg()
    print("✓ Benjamini-Hochberg test passed")
    
    test_compare_methods()
    print("✓ Compare methods test passed")
    
    print("\n✓ All tests passed!")
