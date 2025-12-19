"""
Quick Start Example

A minimal example showing the most common use case:
analyzing a simple A/B test with conversion rates.
"""

import numpy as np
from ab_test_framework import (
    SignificanceTest,
    EffectSizeCalculator,
    ResultVisualizer
)

print("Quick A/B Test Analysis")
print("=" * 50)

# Your experiment data
conversions_control = 100
n_control = 1000
conversions_treatment = 130
n_treatment = 1000

print(f"\nControl: {conversions_control}/{n_control} = {conversions_control/n_control:.2%}")
print(f"Treatment: {conversions_treatment}/{n_treatment} = {conversions_treatment/n_treatment:.2%}")

# Step 1: Test for significance
print("\n1. Significance Test")
print("-" * 50)
tester = SignificanceTest()
result = tester.proportions_test(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Significant at α=0.05? {result['significant']}")
print(f"Relative lift: {result['rates']['relative_lift']:.1%}")

# Step 2: Calculate effect size with confidence interval
print("\n2. Effect Size (with 95% CI)")
print("-" * 50)
calc = EffectSizeCalculator()
effect = calc.absolute_difference_ci(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"Absolute difference: {effect['absolute_difference']:.2%}")
print(f"95% CI: [{effect['confidence_interval'][0]:.2%}, {effect['confidence_interval'][1]:.2%}]")

# Step 3: Visualize
print("\n3. Creating Visualization")
print("-" * 50)
viz = ResultVisualizer()
fig = viz.plot_conversion_rates(
    rate_control=result['rates']['control'],
    rate_treatment=result['rates']['treatment'],
    ci_control=effect['confidence_interval'],
    ci_treatment=(
        result['rates']['treatment'] - effect['confidence_interval'][0] + result['rates']['control'],
        result['rates']['treatment'] - effect['confidence_interval'][1] + result['rates']['control']
    ),
    n_control=n_control,
    n_treatment=n_treatment,
    save_path="quick_start_results.png"
)
print("✓ Chart saved as 'quick_start_results.png'")

print("\n" + "=" * 50)
print("Analysis complete!")
