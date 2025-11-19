# Metric Card for Circular Bias Detection (CBD) Integrity Score

## ⚠️ Experimental Status

**This metric is a research prototype and should be considered experimental.** Key limitations:

- **Mathematical foundations**: While based on established statistical methods, the specific combination and thresholds lack rigorous theoretical justification
- **Empirical validation**: Limited validation on real-world datasets; primarily tested on synthetic data
- **Threshold calibration**: Risk level thresholds (30, 60) are heuristic and not validated across diverse domains
- **Causal inference**: Measures correlation, not causation; cannot definitively prove circular bias

**Recommendation**: Use as a **screening tool** to identify evaluations warranting further investigation, not as definitive evidence of circular bias.

---

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
  - 0-30: Low risk (evaluation appears statistically sound)
  - 30-60: Moderate risk (some circular dependency detected)
  - 60-100: High risk (significant circular bias detected)
  - ⚠️ **Note**: Thresholds are heuristic guidelines, not validated standards. Interpret in context of your specific domain.
- **`rho_pc`** (`float`): Protocol-Performance correlation (-1 to 1).
  - Measures linear correlation between protocol changes and performance
  - High |ρ_PC| indicates correlation, not necessarily causation
- **`risk_level`** (`str`): "LOW", "MODERATE", or "HIGH".
  - Based on heuristic thresholds; use as screening guidance
- **`recommendation`** (`str`): Actionable guidance based on detected risk level.
- **`psi_score`** (`float`, optional): Parameter stability indicator.
- **`ccs_score`** (`float`, optional): Constraint consistency indicator.

## Mathematical Foundations

For rigorous mathematical definitions, statistical properties, and detailed limitations of each indicator, see:
- **[MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)** - Comprehensive mathematical documentation

### Core Indicators

1. **ρ_PC (Protocol-Performance Correlation)**
   - Pearson correlation between protocol variations and performance scores
   - Range: [-1, 1]; values near ±1 indicate strong correlation
   - **Limitation**: Measures correlation, not causation; sensitive to outliers

2. **PSI (Performance-Structure Independence)** [Optional]
   - Measures parameter stability across evaluation periods
   - Formula: Average absolute difference in parameters over time
   - **Limitation**: Cannot distinguish legitimate improvement from circular bias

3. **CCS (Constraint-Consistency Score)** [Optional]
   - Measures consistency of constraint specifications using coefficient of variation
   - Range: [0, 1]; higher values indicate more consistency
   - **Limitation**: Undefined for zero-mean constraints; assumes continuous constraints

---

## Limitations and Bias

### Critical Limitations

1. **Experimental Status**: This metric is a research prototype, not a validated production tool.

2. **Threshold Validity**: 
   - Risk thresholds (30, 60) are **heuristic**, not empirically validated
   - No cross-domain calibration performed
   - Optimal thresholds likely vary by research area
   - **Recommendation**: Calibrate thresholds for your specific domain

3. **Correlation vs. Causation**:
   - High ρ_PC indicates correlation, **not proof of circular bias**
   - Possible explanations: legitimate improvement, confounding factors, reverse causation
   - **Recommendation**: Use as screening tool, not definitive evidence

4. **Protocol Quantification**:
   - Requires user to quantify protocol variations (subjective)
   - Quality of results depends on quality of quantification
   - No standard method for quantifying "protocol change magnitude"

5. **Minimum Data Requirements**:
   - Requires ≥3 evaluation rounds (minimum)
   - Reliable results need ≥10 rounds
   - Small sample sizes produce high variance estimates

6. **Statistical Assumptions**:
   - Assumes linear relationships (may miss non-linear patterns)
   - Assumes independence of observations (often violated in time series)
   - Assumes bivariate normality for significance testing
   - Sensitive to outliers and measurement noise

7. **Limited Validation**:
   - Primarily tested on synthetic data with known bias patterns
   - Limited real-world validation across diverse domains
   - Detection rates on real-world circular bias are **unknown**

### False Positive Scenarios

- **Legitimate research progress**: Natural co-evolution of methods and protocols
- **Exploratory research**: Intentional protocol exploration in early-stage research
- **External factors**: Hardware improvements, dataset updates affecting both protocol and performance
- **Reactive adjustments**: Poor performance motivating protocol changes (reverse causation)

### False Negative Scenarios

- **Non-linear bias**: Circular bias through non-linear protocol-performance relationships
- **Discrete protocols**: Bias through categorical protocol choices not captured by correlation
- **Dataset selection**: Circular bias through dataset curation rather than protocol tuning
- **Time-lagged effects**: Protocol changes at time t affecting performance at t+k

### When NOT to Use

1. **Exploratory research**: Early-stage research with intentional protocol exploration
2. **Small sample sizes**: T < 5 provides unreliable estimates
3. **Qualitative protocols**: When protocol changes cannot be meaningfully quantified
4. **Single evaluation**: CBD requires multiple evaluation periods
5. **Categorical protocols**: When protocols are primarily categorical rather than continuous

---

## Validation Status

### Current Validation

- ✅ **Synthetic data**: Tested on synthetic datasets with injected bias patterns
- ⚠️ **Real-world data**: Limited validation on actual research evaluations
- ❌ **Cross-domain**: No systematic cross-domain validation
- ❌ **Threshold calibration**: Thresholds not empirically optimized

### What We Know

- CBD can detect **known, injected bias** in synthetic scenarios (as expected)
- Correlation-based approach is theoretically sound for **screening**
- Mathematical foundations are based on established statistical methods

### What We Don't Know

- **False positive rate** in real-world scenarios
- **False negative rate** for sophisticated circular bias
- **Optimal thresholds** for different research domains
- **Generalization** across diverse evaluation contexts
- **Comparative effectiveness** vs. other bias detection methods

### Call for Contribution

We welcome community contributions:
- Real-world test cases with ground truth labels
- Domain-specific threshold calibration studies
- Comparative evaluations vs. alternative methods
- Extensions to handle categorical protocols and non-linear relationships

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
