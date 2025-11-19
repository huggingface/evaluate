# Metric Card for Circular Bias Detection (CBD) Integrity Score

## Metric Description

The **Circular Bias Detection (CBD) Integrity Score** is a meta-evaluation metric that measures the statistical integrity of AI evaluation processes. Unlike traditional metrics that measure model performance (e.g., accuracy, F1, BLEU), CBD measures whether the evaluation process itself is trustworthy and free from circular reasoning bias.

**Circular reasoning bias** occurs when evaluation results become artificially inflated through iterative protocol adjustments that optimize for benchmark performance rather than true model generalization. This is a critical but often overlooked issue in AI evaluation.

## How to Use

### Basic Usage

```python
import evaluate
import numpy as np

# Load the metric
cbd_metric = evaluate.load("circular_bias_integrity")

# Example: 5 evaluation rounds with increasing performance and protocol changes
performance_scores = [0.85, 0.87, 0.91, 0.89, 0.93]
protocol_variations = [0.1, 0.15, 0.25, 0.20, 0.30]

# Compute CBD score
results = cbd_metric.compute(
    performance_scores=performance_scores,
    protocol_variations=protocol_variations
)

print(f"CBD Score: {results['cbd_score']:.1f}")
print(f"ρ_PC: {results['rho_pc']:.3f}")
print(f"Risk Level: {results['risk_level']}")
print(f"Recommendation: {results['recommendation']}")
```

### Advanced Usage with Full Matrix Data

```python
import evaluate
import numpy as np

cbd_metric = evaluate.load("circular_bias_integrity")

# Performance across 5 time periods for 3 algorithms
performance_matrix = np.array([
    [0.85, 0.78, 0.82],
    [0.87, 0.80, 0.84],
    [0.91, 0.84, 0.88],
    [0.89, 0.82, 0.86],
    [0.93, 0.86, 0.90]
])

# Constraint specifications (e.g., batch_size, learning_rate)
constraint_matrix = np.array([
    [512, 0.001],
    [550, 0.0015],
    [600, 0.002],
    [580, 0.0018],
    [620, 0.0022]
])

# Compute all indicators
results = cbd_metric.compute(
    performance_scores=performance_matrix.mean(axis=1).tolist(),
    protocol_variations=[0.1, 0.15, 0.25, 0.20, 0.30],
    performance_matrix=performance_matrix,
    constraint_matrix=constraint_matrix,
    return_all_indicators=True
)

print(f"ρ_PC (Protocol-Performance Correlation): {results['rho_pc']:.3f}")
print(f"PSI (Performance-Structure Independence): {results['psi_score']:.3f}")
print(f"CCS (Constraint-Consistency Score): {results['ccs_score']:.3f}")
```

### Inputs

- **`performance_scores`** (`list` of `float`): Performance scores across multiple evaluation rounds. Minimum 3 rounds required.
- **`protocol_variations`** (`list` of `float`): Quantified protocol variation magnitudes for each evaluation round.
- **`performance_matrix`** (`array-like`, optional): Shape (T, K) for detailed multi-algorithm tracking.
- **`constraint_matrix`** (`array-like`, optional): Shape (T, p) for constraint specification tracking.
- **`return_all_indicators`** (`bool`, optional): Return all three indicators (ρ_PC, PSI, CCS). Default: `False`.

### Output Values

- **`cbd_score`** (`float`): Overall integrity score (0-100). Higher = more bias detected.
  - 0-30: Low risk
  - 30-60: Moderate risk  
  - 60-100: High risk
- **`rho_pc`** (`float`): Protocol-Performance correlation (-1 to 1).
- **`risk_level`** (`str`): "LOW", "MODERATE", or "HIGH".
- **`recommendation`** (`str`): Actionable guidance.
- **`psi_score`** (`float`, optional): Parameter stability indicator.
- **`ccs_score`** (`float`, optional): Constraint consistency indicator.

## Limitations and Bias

### Limitations

1. **Minimum Data Requirements**: Requires at least 3 evaluation rounds for reliable correlation analysis. More rounds (5-10+) provide more robust results.

2. **Protocol Quantification**: Users must quantify protocol variations, which can be subjective. Consider using normalized measures (e.g., percentage change in hyperparameters).

3. **Correlation ≠ Causation**: High ρ_PC indicates correlation between protocol changes and performance, but doesn't prove causation. Manual inspection is recommended.

4. **Simplified MVP**: This initial version focuses on ρ_PC as the primary indicator. Full CBD framework includes bootstrap confidence intervals and adaptive thresholds (available in the standalone library).

### Bias Considerations

- **False Positives**: Natural performance improvements during model development may be flagged as circular bias if correlated with protocol changes.
- **False Negatives**: Sophisticated circular bias (e.g., through dataset selection) may not be detected if protocol variations aren't properly quantified.

## Citation

```bibtex
@article{zhang2025circular,
  title={Circular Bias Detection: A Comprehensive Statistical Framework for Detecting Circular Reasoning Bias in AI Algorithm Evaluation},
  author={Zhang, Hongping},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025},
  note={Software available at: https://github.com/hongping-zh/circular-bias-detection}
}
```

## Further References

- **GitHub Repository**: [hongping-zh/circular-bias-detection](https://github.com/hongping-zh/circular-bias-detection)
- **Software DOI**: [10.5281/zenodo.17201032](https://doi.org/10.5281/zenodo.17201032)
- **Dataset DOI**: [10.5281/zenodo.17196639](https://doi.org/10.5281/zenodo.17196639)
- **Live Demo**: [Try Sleuth (CBD Web App)](https://is.gd/check_sleuth)

## Acknowledgements

This metric implements the Circular Bias Detection (CBD) framework developed by Hongping Zhang. The framework addresses a critical gap in AI evaluation methodology by providing quantitative tools for assessing evaluation integrity.

**Slogan**: *Ensuring your evaluation is trustworthy. Stop circular reasoning in AI benchmarks.*
