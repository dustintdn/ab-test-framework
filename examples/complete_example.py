"""
Complete A/B Test Analysis Example

This script demonstrates all features of the framework with
a realistic A/B test scenario.
"""

import numpy as np
from ab_test_framework import (
    PowerAnalyzer,
    SignificanceTest,
    EffectSizeCalculator,
    MultipleTestingCorrection,
    ResultVisualizer
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("A/B TEST ANALYSIS FRAMEWORK - COMPLETE EXAMPLE")
print("=" * 70)

# ============================================================================
# SCENARIO: Testing a new website checkout flow
# 
# We want to test if a simplified checkout process increases conversion rate.
# Control: Current checkout (3 steps)
# Treatment: New simplified checkout (1 step)
# ============================================================================

# ============================================================================
# STEP 1: POWER ANALYSIS (Before running the experiment)
# ============================================================================
print("\n" + "="*70)
print("STEP 1: POWER ANALYSIS")
print("="*70)
print("\nBefore we start the experiment, let's determine required sample size.\n")

analyzer = PowerAnalyzer(alpha=0.05)

# Current conversion rate is 10%, we want to detect a 2 percentage point lift
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.10,
    minimum_detectable_effect=0.02,  # 2 percentage points
    power=0.8,
    ratio=1.0  # Equal size groups
)

print(f"Baseline conversion rate: {sample_size['assumptions']['baseline_rate']:.1%}")
print(f"Minimum detectable effect: {sample_size['assumptions']['minimum_detectable_effect']:.1%}")
print(f"Desired power: {sample_size['assumptions']['power']:.0%}")
print(f"\nRequired sample size per group: {sample_size['n_control']:,}")
print(f"Total sample size needed: {sample_size['total_sample_size']:,}")

# ============================================================================
# STEP 2: RUN THE EXPERIMENT (Simulate data)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: EXPERIMENT EXECUTION")
print("="*70)
print("\nRunning experiment with calculated sample sizes...\n")

# Simulate experiment data
n_control = sample_size['n_control']
n_treatment = sample_size['n_treatment']

# Control group: 10% conversion rate
conversions_control = np.random.binomial(n_control, 0.10)

# Treatment group: 12% conversion rate (actual effect in our simulation)
conversions_treatment = np.random.binomial(n_treatment, 0.12)

print(f"Control group: {conversions_control} conversions out of {n_control} ({conversions_control/n_control:.2%})")
print(f"Treatment group: {conversions_treatment} conversions out of {n_treatment} ({conversions_treatment/n_treatment:.2%})")

# ============================================================================
# STEP 3: SIGNIFICANCE TESTING
# ============================================================================
print("\n" + "="*70)
print("STEP 3: SIGNIFICANCE TESTING")
print("="*70)

tester = SignificanceTest(alpha=0.05)

# Parametric test (traditional approach)
print("\n--- Parametric Test (Two-proportion z-test) ---")
result_parametric = tester.proportions_test(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"P-value: {result_parametric['p_value']:.4f}")
print(f"Significant: {result_parametric['significant']}")
print(f"Interpretation: {result_parametric['interpretation']}")
print(f"\nControl rate: {result_parametric['rates']['control']:.2%}")
print(f"Treatment rate: {result_parametric['rates']['treatment']:.2%}")
print(f"Absolute difference: {result_parametric['rates']['difference']:.2%}")
print(f"Relative lift: {result_parametric['rates']['relative_lift']:.1%}")

# Bootstrap test (non-parametric approach)
print("\n--- Bootstrap Test (Non-parametric) ---")

# Create binary arrays (1 = conversion, 0 = no conversion)
control_data = np.concatenate([
    np.ones(conversions_control),
    np.zeros(n_control - conversions_control)
])
treatment_data = np.concatenate([
    np.ones(conversions_treatment),
    np.zeros(n_treatment - conversions_treatment)
])

result_bootstrap = tester.bootstrap_test(
    values_control=control_data,
    values_treatment=treatment_data,
    n_bootstrap=10000,
    statistic="proportion",
    random_seed=42
)

print(f"P-value: {result_bootstrap['p_value']:.4f}")
print(f"Significant: {result_bootstrap['significant']}")
print(f"Interpretation: {result_bootstrap['interpretation']}")

# ============================================================================
# STEP 4: EFFECT SIZE ESTIMATION
# ============================================================================
print("\n" + "="*70)
print("STEP 4: EFFECT SIZE ESTIMATION")
print("="*70)

calc = EffectSizeCalculator(confidence_level=0.95)

# Absolute difference with confidence interval
print("\n--- Absolute Difference ---")
abs_diff = calc.absolute_difference_ci(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"Absolute difference: {abs_diff['absolute_difference']:.2%}")
print(f"95% CI: [{abs_diff['confidence_interval'][0]:.2%}, {abs_diff['confidence_interval'][1]:.2%}]")
print(f"Interpretation: {abs_diff['interpretation']}")

# Relative lift with confidence interval
print("\n--- Relative Lift ---")
rel_lift = calc.relative_lift_ci(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"Relative lift: {rel_lift['relative_lift']:.1%}")
print(f"95% CI: [{rel_lift['confidence_interval'][0]:.1%}, {rel_lift['confidence_interval'][1]:.1%}]")
print(f"Interpretation: {rel_lift['interpretation']}")

# ============================================================================
# STEP 5: POST-HOC POWER ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: POST-HOC POWER ANALYSIS")
print("="*70)
print("\nEvaluating whether our experiment had sufficient power...\n")

power_result = analyzer.calculate_power(
    n_control=n_control,
    n_treatment=n_treatment,
    baseline_rate=conversions_control/n_control,
    treatment_rate=conversions_treatment/n_treatment
)

print(f"Actual power: {power_result['power']:.1%}")
print(f"Interpretation: {power_result['interpretation']}")
print(f"Effect size: {power_result['effect_size']:.2%} (absolute)")
print(f"Relative effect: {power_result['effect_size_relative']:.1%}")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 6: CREATING VISUALIZATIONS")
print("="*70)

viz = ResultVisualizer()

# Plot conversion rates
print("\nCreating conversion rate comparison plot...")
fig1 = viz.plot_conversion_rates(
    rate_control=result_parametric['rates']['control'],
    rate_treatment=result_parametric['rates']['treatment'],
    ci_control=abs_diff['confidence_interval'],
    ci_treatment=(
        result_parametric['rates']['treatment'] - abs_diff['confidence_interval'][0] + result_parametric['rates']['control'],
        result_parametric['rates']['treatment'] - abs_diff['confidence_interval'][1] + result_parametric['rates']['control']
    ),
    n_control=n_control,
    n_treatment=n_treatment,
    title="Checkout Flow A/B Test Results",
    save_path="conversion_rates.png"
)
print("✓ Saved: conversion_rates.png")

# Plot effect size
print("Creating effect size visualization...")
fig2 = viz.plot_effect_size(
    effect_size=abs_diff['absolute_difference'],
    ci_lower=abs_diff['confidence_interval'][0],
    ci_upper=abs_diff['confidence_interval'][1],
    metric_name="Absolute Conversion Rate Difference",
    title="Effect Size: Absolute Difference in Conversion Rate",
    save_path="effect_size.png"
)
print("✓ Saved: effect_size.png")

# ============================================================================
# BONUS: MULTIPLE TESTING SCENARIO
# ============================================================================
print("\n" + "="*70)
print("BONUS: MULTIPLE TESTING CORRECTION")
print("="*70)
print("\nSuppose we tested 5 different metrics in this experiment:\n")

# Simulate p-values for multiple metrics
test_names = ["Conversion", "Revenue per User", "Time on Site", "Cart Adds", "Page Views"]
p_values = [0.01, 0.04, 0.08, 0.12, 0.25]  # Simulated p-values

corrector = MultipleTestingCorrection()

# Compare correction methods
comparison = corrector.compare_methods(p_values, alpha=0.05)

print("P-values for each metric:")
for name, p in zip(test_names, p_values):
    print(f"  {name}: {p:.3f}")

print(f"\nUncorrected significant tests: {comparison['summary']['uncorrected_significant']}")
print(f"Bonferroni: {comparison['summary']['bonferroni_significant']} significant")
print(f"Bonferroni-Holm: {comparison['summary']['bonferroni_holm_significant']} significant")
print(f"Benjamini-Hochberg (FDR): {comparison['summary']['benjamini_hochberg_significant']} significant")
print(f"\nRecommendation: {comparison['recommendation']}")

# Visualize correction
print("\nCreating multiple testing visualization...")
fig3 = viz.plot_multiple_tests(
    test_names=test_names,
    p_values=p_values,
    adjusted_p_values=comparison['benjamini_hochberg']['adjusted_p_values'],
    alpha=0.05,
    title="Multiple Testing: Benjamini-Hochberg Correction",
    save_path="multiple_testing.png"
)
print("✓ Saved: multiple_testing.png")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)

print(f"""
EXPERIMENT RESULTS:
- Control conversion rate: {result_parametric['rates']['control']:.2%}
- Treatment conversion rate: {result_parametric['rates']['treatment']:.2%}
- Absolute lift: {abs_diff['absolute_difference']:.2%}
- Relative lift: {rel_lift['relative_lift']:.1%}
- Statistical significance: {'YES' if result_parametric['significant'] else 'NO'} (p={result_parametric['p_value']:.4f})
- Actual power: {power_result['power']:.1%}

RECOMMENDATION:
""")

if result_parametric['significant'] and power_result['power'] >= 0.8:
    print("✓ The treatment shows a statistically significant improvement with good power.")
    print("  RECOMMEND: Implement the simplified checkout flow.")
elif result_parametric['significant'] and power_result['power'] < 0.8:
    print("⚠ Significant result but power is below 80%.")
    print("  RECOMMEND: Consider running a longer test to confirm the effect.")
elif not result_parametric['significant'] and power_result['power'] >= 0.8:
    print("✗ No significant effect detected with adequate power.")
    print("  RECOMMEND: Keep current checkout flow or test a different variant.")
else:
    print("⚠ No significant effect and insufficient power.")
    print("  RECOMMEND: Run a larger experiment before making a decision.")

print("\n" + "="*70)
print("Analysis complete! Check the saved visualizations.")
print("="*70)
