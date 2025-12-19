# A/B Test Analysis Framework

A simple, readable Python Framework for analyzing A/B tests and experiments.

## Features

- **Power Analysis**: Determine required sample sizes before running experiments
- **Significance Testing**: Both parametric and bootstrap-based tests
- **Effect Size Estimation**: Calculate practical significance with confidence intervals
- **Multiple Testing Correction**: Bonferroni, Bonferroni-Holm, and Benjamini-Hochberg methods
- **Clear Visualizations**: Publication-ready plots for presenting results

## Project Structure

```
ab_test_framework/
├── ab_test_framework/          # Main package directory
│   ├── __init__.py            # Package initialization and exports
│   ├── power_analysis.py      # Power and sample size calculations
│   ├── significance_tests.py  # Statistical significance tests
│   ├── effect_size.py         # Effect size calculations with CIs
│   ├── corrections.py         # Multiple testing corrections
│   └── visualizations.py      # Publication-ready visualizations
│
├── examples/                   # Usage examples
│   ├── quick_start.py         # Minimal 5-minute example
│   └── complete_example.py    # Full workflow demonstration
│
├── tests/
│   └── test_basic.py          # Unit tests for all modules
│
├── README.md    
├── requirements.txt            
└── setup.py                    # Package installation script
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Add the package to your Python path or install in development mode
pip install -e .
```

## Quick Start

```python
from ab_test_framework import SignificanceTest, EffectSizeCalculator, ResultVisualizer

# Your experiment data
conversions_control = 100
n_control = 1000
conversions_treatment = 130
n_treatment = 1000

# Test for significance
tester = SignificanceTest()
result = tester.proportions_test(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Relative lift: {result['rates']['relative_lift']:.1%}")

# Calculate effect size
calc = EffectSizeCalculator()
effect = calc.absolute_difference_ci(
    conversions_control=conversions_control,
    n_control=n_control,
    conversions_treatment=conversions_treatment,
    n_treatment=n_treatment
)

print(f"Effect: {effect['absolute_difference']:.2%}")
print(f"95% CI: {effect['confidence_interval']}")
```

## Key Concepts

### Power Analysis

**Before you run an experiment**, use power analysis to determine how many samples you need:

```python
from ab_test_framework import PowerAnalyzer

analyzer = PowerAnalyzer(alpha=0.05)

# Calculate required sample size
result = analyzer.calculate_sample_size(
    baseline_rate=0.10,              # Current conversion rate
    minimum_detectable_effect=0.02,  # Smallest effect you care about
    power=0.8,                        # Standard is 80%
    ratio=1.0                         # Equal group sizes
)

print(f"Need {result['n_control']} samples per group")
```

**After your experiment**, check if you had sufficient power:

```python
power = analyzer.calculate_power(
    n_control=1000,
    n_treatment=1000,
    baseline_rate=0.10,
    treatment_rate=0.12
)

print(f"Your experiment had {power['power']:.1%} power")
```

### Significance Testing

#### Parametric Tests (Traditional)

For conversion rates:
```python
from ab_test_framework import SignificanceTest

tester = SignificanceTest(alpha=0.05)
result = tester.proportions_test(
    conversions_control=100,
    n_control=1000,
    conversions_treatment=130,
    n_treatment=1000
)
```

For continuous metrics (revenue, time, etc.):
```python
result = tester.means_test(
    values_control=control_revenue,
    values_treatment=treatment_revenue
)
```

#### Bootstrap Tests (Non-parametric)

Use when your data doesn't meet parametric assumptions (e.g., highly skewed):

```python
result = tester.bootstrap_test(
    values_control=control_data,
    values_treatment=treatment_data,
    n_bootstrap=10000,
    statistic="median",  # or "mean" or "proportion"
    random_seed=42
)
```

### Effect Sizes

Statistical significance tells you *if* there's an effect. Effect sizes tell you *how big* it is.

#### Cohen's d (for continuous metrics)

```python
from ab_test_framework import EffectSizeCalculator

calc = EffectSizeCalculator(confidence_level=0.95)

result = calc.cohens_d(
    values_control=control_revenue,
    values_treatment=treatment_revenue
)

print(f"Cohen's d: {result['cohens_d']:.3f}")
print(f"95% CI: {result['confidence_interval']}")
```

**Interpretation**: 0.2 = small, 0.5 = medium, 0.8 = large effect

#### Absolute and Relative Lift (for conversion rates)

```python
# Absolute difference (e.g., +2 percentage points)
abs_result = calc.absolute_difference_ci(
    conversions_control=100,
    n_control=1000,
    conversions_treatment=130,
    n_treatment=1000
)

# Relative lift (e.g., +30% improvement)
rel_result = calc.relative_lift_ci(
    conversions_control=100,
    n_control=1000,
    conversions_treatment=130,
    n_treatment=1000
)
```

### Multiple Testing Correction

When testing multiple metrics, you need to adjust for multiple comparisons:

```python
from ab_test_framework import MultipleTestingCorrection

corrector = MultipleTestingCorrection()

# Your p-values from multiple tests
p_values = [0.01, 0.04, 0.08, 0.12, 0.25]

# Compare different correction methods
comparison = corrector.compare_methods(p_values, alpha=0.05)

print(f"Bonferroni: {comparison['summary']['bonferroni_significant']} significant")
print(f"Benjamini-Hochberg: {comparison['summary']['benjamini_hochberg_significant']} significant")
```

**When to use which method:**
- **Bonferroni**: Most conservative. Use when false positives are very costly.
- **Bonferroni-Holm**: Good balance. Less conservative than Bonferroni.
- **Benjamini-Hochberg**: Controls False Discovery Rate. Best for exploratory analysis.

### Visualizations

Create publication-ready plots:

```python
from ab_test_framework import ResultVisualizer

viz = ResultVisualizer()

# Conversion rate comparison
fig = viz.plot_conversion_rates(
    rate_control=0.10,
    rate_treatment=0.13,
    ci_control=(0.08, 0.12),
    ci_treatment=(0.11, 0.15),
    n_control=1000,
    n_treatment=1000,
    title="A/B Test Results",
    save_path="results.png"
)

# Effect size visualization
fig = viz.plot_effect_size(
    effect_size=0.03,
    ci_lower=0.01,
    ci_upper=0.05,
    metric_name="Conversion Rate Difference"
)

# Distribution comparison
fig = viz.plot_distributions(
    values_control=control_data,
    values_treatment=treatment_data,
    metric_name="Revenue per User"
)
```

## Examples

See the `examples/` directory for complete working examples:

- **`quick_start.py`**: Minimal example for analyzing a simple A/B test
- **`complete_example.py`**: Full workflow from power analysis to final recommendation

Run an example:
```bash
cd examples
python complete_example.py
```

## Best Practices

### 1. Always Do Power Analysis First
Don't run an experiment without knowing how many samples you need. Underpowered experiments waste time and resources.

### 2. Pre-register Your Analysis
Decide on your primary metric and analysis approach *before* looking at results. This prevents p-hacking.

### 3. Report Effect Sizes, Not Just P-values
A tiny effect can be statistically significant with enough data. Effect sizes tell you if it's practically meaningful.

### 4. Use Confidence Intervals
They provide more information than a binary "significant/not significant" decision.

### 5. Correct for Multiple Testing
If you're testing multiple metrics, use appropriate corrections to avoid false positives.

### 6. Check Your Assumptions
- For parametric tests: Are distributions roughly normal? Are variances similar?
- If assumptions don't hold, use bootstrap methods instead.

### 7. Consider Practical Significance
A statistically significant 0.1% lift might not be worth implementing. Think about the business impact.

## Understanding the Output

All functions return dictionaries with clear keys. Here's what to focus on:

### Significance Tests
- `p_value`: Probability of seeing this result if there's no true effect
- `significant`: Is p < α? (typically α = 0.05)
- `interpretation`: Plain English explanation

### Effect Sizes
- `effect_size` or `absolute_difference`: Magnitude of the effect
- `confidence_interval`: Range of plausible values
- `interpretation`: Helps you understand the practical importance

### Power Analysis
- `power`: Probability of detecting a true effect (aim for ≥80%)
- `n_control`, `n_treatment`: Required sample sizes
- `interpretation`: Whether power is adequate

## Common Pitfalls to Avoid

1. **Peeking at results early**: This inflates false positive rates. Wait until you have your planned sample size.

2. **Testing many metrics without correction**: If you test 20 things at α=0.05, you expect 1 false positive even if nothing works.

3. **Stopping early because "it's significant"**: You might have just gotten lucky. Stick to your planned sample size.

4. **Confusing statistical and practical significance**: A tiny effect can be statistically significant with large samples but not worth implementing.

5. **Ignoring confidence intervals**: They tell you the precision of your estimate and help assess practical significance.

## When to Use What

| Scenario | Use This |
|----------|----------|
| Planning an experiment | `PowerAnalyzer.calculate_sample_size()` |
| Testing conversion rates | `SignificanceTest.proportions_test()` |
| Testing continuous metrics | `SignificanceTest.means_test()` |
| Skewed data or outliers | `SignificanceTest.bootstrap_test()` |
| Understanding effect magnitude | `EffectSizeCalculator` methods |
| Testing multiple metrics | `MultipleTestingCorrection` |
| Checking post-hoc power | `PowerAnalyzer.calculate_power()` |


## References

- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*
- VanderPlas, J. (2016). *Python Data Science Handbook*
- Wasserstein, R. L., & Lazar, N. A. (2016). "The ASA Statement on p-Values"
