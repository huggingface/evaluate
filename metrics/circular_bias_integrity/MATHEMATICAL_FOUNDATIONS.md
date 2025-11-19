# Mathematical Foundations of CBD Framework

**Status**: Research Prototype  
**Last Updated**: 2025-11-19

---

## Overview

This document provides rigorous mathematical definitions, statistical properties, and limitations of the three core indicators in the Circular Bias Detection (CBD) framework. These definitions are essential for understanding the theoretical basis and practical constraints of the metric.

---

## 1. PSI (Performance-Structure Independence)

### 1.1 Mathematical Definition

**Purpose**: Measure parameter stability across evaluation periods to detect iterative protocol adjustments.

**Formal Definition**:

$$\text{PSI} = \frac{1}{K}\sum_{k=1}^{K} \text{PSI}_k$$

where for each algorithm $k$:

$$\text{PSI}_k = \frac{1}{T-1}\sum_{t=1}^{T-1}|\theta_{k,t+1} - \theta_{k,t}|$$

**Notation**:
- $K$: Number of algorithms being evaluated
- $T$: Number of evaluation time periods (must be $T \geq 2$)
- $\theta_{k,t}$: Parameter value (or performance proxy) for algorithm $k$ at time $t$
- $|\cdot|$: Absolute value (L1 norm for scalar parameters)

### 1.2 Statistical Properties

**Expected Value**:
Under the null hypothesis of no systematic parameter drift:
$$E[\text{PSI}] \approx \sigma_{\epsilon} \sqrt{\frac{2}{\pi}}$$
where $\sigma_{\epsilon}$ is the standard deviation of measurement noise.

**Variance**:
$$\text{Var}(\text{PSI}) \approx \frac{\sigma_{\epsilon}^2}{T-1}\left(1 - \frac{2}{\pi}\right)$$

**Distribution**:
- For large $T$, PSI approximately follows a folded normal distribution
- For small $T$ (< 10), distribution is highly skewed
- Assumes independence of consecutive measurements (often violated in practice)

### 1.3 Sensitivity Analysis

**Sample Size Sensitivity**:
- $T < 3$: Unreliable (insufficient data for trend detection)
- $3 \leq T < 10$: Moderate reliability (high variance)
- $T \geq 10$: Good reliability (variance decreases as $O(1/T)$)

**Outlier Sensitivity**:
- PSI uses L1 norm (absolute difference), which is more robust than L2 norm
- However, single outliers can still significantly affect results when $T$ is small
- Recommendation: Use robust estimators (e.g., median absolute deviation) for $T < 10$

**Measurement Noise**:
$$\text{Signal-to-Noise Ratio} = \frac{|\Delta\theta_{\text{true}}|}{\sigma_{\epsilon}}$$
- SNR < 1: PSI dominated by noise
- SNR ≥ 3: PSI reliably detects parameter changes

### 1.4 Limitations and Assumptions

**Assumptions**:
1. Parameters are continuous and comparable across time periods
2. Measurement errors are independent and identically distributed (i.i.d.)
3. No external factors systematically affect parameter values
4. Parameter changes are independent across algorithms

**Known Limitations**:
1. **Cannot distinguish legitimate vs. circular parameter changes**: High PSI may indicate either:
   - Circular bias (iterative protocol tuning)
   - Legitimate model improvement (algorithm evolution)
   - External factors (hardware changes, dataset updates)

2. **Assumes stationarity**: Does not account for expected parameter drift in rapidly evolving research areas

3. **Scale dependency**: PSI magnitude depends on parameter scale. Recommendation: Normalize parameters before computing PSI

4. **Temporal correlation**: Consecutive evaluations are often correlated, violating independence assumption

**False Positive Scenarios**:
- Active research areas with rapid algorithm development
- Exploratory phases with intentional parameter sweeps
- Hardware or software environment changes

**False Negative Scenarios**:
- Small, incremental parameter adjustments (below measurement noise)
- Parameter changes in orthogonal dimensions (not captured by scalar proxy)
- Circular bias through dataset selection rather than parameter tuning

---

## 2. CCS (Constraint-Consistency Score)

### 2.1 Mathematical Definition

**Purpose**: Measure consistency of constraint specifications across evaluation periods.

**Formal Definition**:

$$\text{CCS} = \frac{1}{p}\sum_{j=1}^{p} \text{CCS}_j$$

where for each constraint type $j$:

$$\text{CCS}_j = \begin{cases}
\frac{1}{1 + CV_j} & \text{if } \mu_j \neq 0 \text{ and } \sigma_j > 0 \\
1 & \text{if } \sigma_j = 0 \text{ (constant constraint)} \\
0 & \text{if } \mu_j = 0 \text{ (undefined CV)}
\end{cases}$$

where the coefficient of variation is:

$$CV_j = \frac{\sigma_j}{|\mu_j|}$$

**Notation**:
- $p$: Number of constraint types (e.g., batch size, learning rate, data split ratio)
- $T$: Number of evaluation time periods
- $c_{j,t}$: Value of constraint $j$ at time $t$
- $\mu_j = \frac{1}{T}\sum_{t=1}^{T} c_{j,t}$: Mean of constraint $j$
- $\sigma_j = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(c_{j,t} - \mu_j)^2}$: Standard deviation of constraint $j$

### 2.2 Why Coefficient of Variation?

**Rationale**:
1. **Scale invariance**: CV is dimensionless, allowing comparison across constraints with different units
   - Example: Can compare consistency of batch size (range: 16-512) with learning rate (range: 0.0001-0.01)

2. **Interpretability**: 
   - $CV = 0$: Perfect consistency (no variation)
   - $CV < 0.1$: High consistency (< 10% relative variation)
   - $CV > 1$: Low consistency (variation exceeds mean)

3. **Standard statistical measure**: Well-established in quality control and reliability engineering

4. **Monotonic transformation**: $\text{CCS}_j = \frac{1}{1+CV_j}$ maps CV to [0,1] range with intuitive interpretation

**Transformation Properties**:
$$\lim_{CV_j \to 0} \text{CCS}_j = 1 \quad \text{(perfect consistency)}$$
$$\lim_{CV_j \to \infty} \text{CCS}_j = 0 \quad \text{(no consistency)}$$

### 2.3 Applicability to Different Constraint Types

**Continuous Constraints** (e.g., learning rate, dropout rate):
- ✅ **Well-suited**: CV directly measures relative variability
- ⚠️ **Caution**: Assumes constraints are on ratio scale (meaningful zero point)

**Discrete Constraints** (e.g., batch size, number of layers):
- ✅ **Acceptable**: CV still meaningful if values span reasonable range
- ⚠️ **Caution**: For small discrete sets (e.g., {16, 32, 64}), CV may overestimate inconsistency

**Categorical Constraints** (e.g., optimizer type, activation function):
- ❌ **Not applicable**: CV undefined for nominal categories
- 🔄 **Alternative**: Use entropy or mode frequency for categorical constraints

**Mixed Constraints**:
- Compute CCS separately for continuous and discrete subsets
- Weight by importance or use separate reporting

### 2.4 Statistical Properties

**Expected Value** (under random constraint selection):
$$E[\text{CCS}] \approx \frac{1}{1 + E[CV]}$$

For uniform random constraints on $[a, b]$:
$$E[CV] = \frac{1}{\sqrt{3}} \approx 0.577$$
$$E[\text{CCS}] \approx 0.634$$

**Variance**:
$$\text{Var}(\text{CCS}) \approx \frac{\text{Var}(CV)}{(1 + E[CV])^4}$$

**Distribution**:
- CCS is bounded in [0, 1]
- Distribution depends on underlying constraint distribution
- For normally distributed constraints, CCS follows a transformed inverse gamma distribution

### 2.5 Limitations and Assumptions

**Assumptions**:
1. Constraints are continuous or ordinal (meaningful ordering)
2. Constraint values are on ratio or interval scale
3. Mean constraint value is non-zero and meaningful
4. Constraint changes are independent across types

**Known Limitations**:

1. **Zero-mean constraints**: 
   - Problem: CV undefined when $\mu_j = 0$
   - Current handling: Set $\text{CCS}_j = 0$ (conservative)
   - Better approach: Use alternative measures (e.g., median absolute deviation)

2. **Near-zero mean constraints**:
   - Problem: CV becomes unstable and arbitrarily large
   - Example: Learning rate oscillating around 0.0001 ± 0.00005 gives $CV = 0.5$, but around 0.00001 ± 0.000005 gives $CV = 0.5$ (same relative variation)
   - Recommendation: Use absolute variation for constraints with $|\mu_j| < \epsilon$

3. **Non-normal distributions**:
   - CV assumes approximate normality
   - For skewed distributions, CV may not accurately reflect consistency
   - Alternative: Use robust measures (e.g., quartile coefficient of dispersion)

4. **Temporal trends**:
   - CCS treats all time periods equally
   - Does not detect systematic trends (e.g., gradual increase in batch size)
   - May miss circular bias that manifests as trends rather than random variation

5. **Constraint interdependencies**:
   - CCS assumes constraints are independent
   - In practice, constraints often covary (e.g., batch size and learning rate)
   - Does not capture multivariate consistency patterns

**False Positive Scenarios**:
- Legitimate exploration of constraint space
- Adaptive constraints (e.g., learning rate schedules)
- Hardware-driven constraint changes (e.g., GPU memory limits)

**False Negative Scenarios**:
- Consistent but biased constraint choices
- Constraints changed in correlated manner (maintaining ratios)
- Circular bias through constraint combinations rather than individual values

---

## 3. ρ_PC (Protocol-Performance Correlation)

### 3.1 Mathematical Definition

**Purpose**: Measure correlation between protocol changes and performance improvements.

**Formal Definition**:

$$\rho_{PC} = \frac{\text{Cov}(P, C)}{\sigma_P \sigma_C}$$

where:
$$\text{Cov}(P, C) = \frac{1}{T-1}\sum_{t=1}^{T}(P_t - \bar{P})(C_t - \bar{C})$$

**Notation**:
- $P_t$: Performance score at time $t$ (aggregated across algorithms if multiple)
- $C_t$: Protocol variation magnitude at time $t$
- $\bar{P} = \frac{1}{T}\sum_{t=1}^{T} P_t$: Mean performance
- $\bar{C} = \frac{1}{T}\sum_{t=1}^{T} C_t$: Mean protocol variation
- $\sigma_P, \sigma_C$: Standard deviations of performance and protocol variation

**Implementation**: Uses Pearson correlation coefficient from `scipy.stats.pearsonr`

### 3.2 Statistical Properties

**Range**: $\rho_{PC} \in [-1, 1]$
- $\rho_{PC} = 1$: Perfect positive correlation (performance increases with protocol changes)
- $\rho_{PC} = 0$: No linear correlation
- $\rho_{PC} = -1$: Perfect negative correlation (performance decreases with protocol changes)

**Statistical Significance**:
Under null hypothesis $H_0: \rho = 0$, the test statistic:
$$t = \rho_{PC}\sqrt{\frac{T-2}{1-\rho_{PC}^2}} \sim t_{T-2}$$

**P-value interpretation**:
- $p < 0.01$: Strong evidence of correlation
- $0.01 \leq p < 0.05$: Moderate evidence
- $p \geq 0.05$: Insufficient evidence (but does not prove independence)

**Power Analysis**:
Minimum sample size for detecting correlation $\rho$ with power $1-\beta$ at significance $\alpha$:
$$T \approx \left(\frac{z_{\alpha/2} + z_{\beta}}{0.5\ln\frac{1+\rho}{1-\rho}}\right)^2 + 3$$

Example: To detect $\rho = 0.5$ with 80% power at $\alpha = 0.05$:
$$T \approx 29$$

### 3.3 Interpretation and Causation

**Critical Distinction**: Correlation ≠ Causation

**High |ρ_PC| may indicate**:
1. ✅ **Circular bias**: Protocol tuned to optimize performance on test set
2. ✅ **Legitimate improvement**: Protocol changes reflect genuine algorithmic advances
3. ✅ **Confounding factors**: External factors affect both protocol and performance
4. ✅ **Reverse causation**: Poor performance motivates protocol changes

**Causal Inference Requirements** (not provided by ρ_PC alone):
- Temporal precedence: Protocol changes must precede performance changes
- Mechanism: Plausible causal pathway from protocol to performance
- No confounders: Other explanations ruled out
- Dose-response: Larger protocol changes → larger performance changes
- Consistency: Pattern holds across different contexts

**Recommendation**: Use ρ_PC as a **screening tool** to identify cases requiring deeper investigation, not as definitive evidence of circular bias.

### 3.4 Sensitivity Analysis

**Sample Size Sensitivity**:
- $T < 3$: Correlation undefined or unreliable
- $3 \leq T < 10$: High variance, wide confidence intervals
- $10 \leq T < 30$: Moderate reliability
- $T \geq 30$: Good reliability for detecting medium-to-large correlations

**Outlier Sensitivity**:
- Pearson correlation is sensitive to outliers
- Single extreme point can dominate correlation
- Recommendation: Use Spearman rank correlation for robustness (not currently implemented)

**Linearity Assumption**:
- Pearson correlation only detects linear relationships
- May miss non-linear protocol-performance relationships
- Example: U-shaped relationship (optimal protocol in middle range)

**Aggregation Effects**:
- When averaging performance across algorithms, individual patterns may be masked
- Recommendation: Compute ρ_PC separately for each algorithm when possible

### 3.5 Limitations and Assumptions

**Assumptions**:
1. Linear relationship between protocol variation and performance
2. Bivariate normal distribution (for significance testing)
3. Homoscedasticity (constant variance)
4. Independence of observations (often violated in time series)
5. No measurement error in protocol variation quantification

**Known Limitations**:

1. **Quantifying protocol variation**:
   - Problem: No standard method to quantify "protocol change magnitude"
   - Current approach: User-provided values (subjective)
   - Impact: ρ_PC validity depends on quality of protocol quantification

2. **Temporal autocorrelation**:
   - Consecutive evaluations are often correlated
   - Violates independence assumption
   - Inflates Type I error rate (false positives)
   - Recommendation: Use time series methods (e.g., Durbin-Watson test)

3. **Multiple testing**:
   - Computing ρ_PC for multiple algorithm pairs increases false positive rate
   - No correction for multiple comparisons currently implemented
   - Recommendation: Apply Bonferroni or FDR correction

4. **Direction ambiguity**:
   - High |ρ_PC| could indicate:
     - Protocol → Performance (circular bias)
     - Performance → Protocol (reactive adjustment)
     - Confounding → Both
   - Cannot distinguish without additional information

5. **Non-linear relationships**:
   - Pearson correlation only captures linear associations
   - May miss important non-linear patterns
   - Alternative: Use mutual information or distance correlation

**False Positive Scenarios**:
- Legitimate co-evolution of methods and protocols
- External factors (e.g., hardware improvements) affecting both
- Natural progression in research (better methods → better protocols)

**False Negative Scenarios**:
- Non-linear protocol-performance relationships
- Circular bias through discrete protocol choices (not captured by continuous correlation)
- Time-lagged effects (protocol change at $t$ affects performance at $t+k$)

---

## 4. Threshold Calibration

### 4.1 Current Thresholds (Heuristic)

**CBD Score** (derived from |ρ_PC| × 100):
- **0-30**: Low risk
- **30-60**: Moderate risk
- **60-100**: High risk

**PSI** (not used in current simplified implementation):
- Threshold: 0.15 (heuristic, domain-dependent)

**CCS** (not used in current simplified implementation):
- Threshold: 0.85 (heuristic, domain-dependent)

### 4.2 Lack of Empirical Validation

**Critical Limitation**: These thresholds are **not validated** through:
- Large-scale empirical studies
- Cross-domain validation
- ROC curve analysis
- Cost-benefit optimization

**Current Status**: Thresholds are **educated guesses** based on:
- Intuition about correlation strength
- Analogies to other fields (e.g., effect size guidelines in psychology)
- Limited synthetic data experiments

### 4.3 Domain-Specific Calibration Needed

**Recommendation**: Users should calibrate thresholds for their specific domain by:

1. **Collect ground truth data**:
   - Known biased evaluations (positive examples)
   - Known unbiased evaluations (negative examples)

2. **Compute ROC curve**:
   - Vary threshold from 0 to 100
   - Plot True Positive Rate vs. False Positive Rate
   - Select threshold based on desired trade-off

3. **Cross-validation**:
   - Use k-fold cross-validation to estimate generalization performance
   - Report confidence intervals for threshold performance

4. **Cost-benefit analysis**:
   - Assign costs to false positives (flagging legitimate research)
   - Assign costs to false negatives (missing circular bias)
   - Optimize threshold to minimize expected cost

### 4.4 Adaptive Thresholds (Future Work)

**Concept**: Adjust thresholds based on data characteristics

**Potential Approaches**:
1. **Quantile-based**: Use empirical distribution of scores
2. **Bayesian**: Update thresholds as more data becomes available
3. **Context-aware**: Adjust based on research area, evaluation type, etc.

**Status**: Not currently implemented. Requires substantial research and validation.

---

## 5. Integrated Framework Limitations

### 5.1 Indicator Independence

**Assumption**: PSI, CCS, and ρ_PC measure independent aspects of circular bias

**Reality**: Indicators are likely correlated:
- High PSI may lead to high |ρ_PC| (parameter instability correlates with performance)
- Low CCS may lead to high |ρ_PC| (inconsistent constraints correlate with performance)

**Impact**: 
- Combining indicators may not provide independent evidence
- "Majority vote" approach may be misleading
- Need multivariate analysis to understand joint distribution

### 5.2 Weighting and Aggregation

**Current Approach**: Simple average or majority vote

**Limitations**:
- Assumes equal importance of indicators (may not be true)
- Does not account for measurement uncertainty
- No principled way to combine conflicting signals

**Better Approaches** (not implemented):
- Weighted combination based on reliability
- Probabilistic framework (e.g., Bayesian network)
- Machine learning classifier trained on labeled data

### 5.3 Temporal Dynamics

**Current Approach**: Static analysis of time series

**Missing**:
- Trend detection (is bias increasing or decreasing?)
- Change point detection (when did circular bias start?)
- Forecasting (will bias continue to increase?)

**Recommendation**: Incorporate time series analysis methods

### 5.4 Causal Inference

**Fundamental Limitation**: CBD framework is **correlational**, not **causal**

**Cannot Answer**:
- Did protocol changes cause performance improvements?
- Would performance have improved without protocol changes?
- What is the counterfactual (performance under different protocols)?

**Causal Methods Needed** (not implemented):
- Randomized controlled trials (RCTs)
- Instrumental variables
- Difference-in-differences
- Regression discontinuity

---

## 6. Recommendations for Users

### 6.1 Interpretation Guidelines

1. **Use as screening tool**: CBD flags potential issues, not definitive proof
2. **Context matters**: Interpret scores in light of research area and practices
3. **Multiple lines of evidence**: Combine CBD with other evaluation integrity checks
4. **Manual inspection**: High scores warrant detailed protocol review
5. **Report uncertainty**: Always report confidence intervals and p-values

### 6.2 Best Practices

1. **Sufficient sample size**: Use $T \geq 10$ evaluation periods when possible
2. **Standardize inputs**: Normalize performance and protocol measures
3. **Document protocol changes**: Maintain detailed logs for post-hoc analysis
4. **Pre-register protocols**: Commit to evaluation protocol before seeing results
5. **Independent validation**: Use held-out test sets not used for protocol tuning

### 6.3 When NOT to Use CBD

1. **Exploratory research**: Early-stage research with intentional protocol exploration
2. **Small sample sizes**: $T < 5$ provides unreliable estimates
3. **Qualitative protocols**: When protocol changes cannot be meaningfully quantified
4. **Single evaluation**: CBD requires multiple evaluation periods

---

## 7. Future Research Directions

### 7.1 Theoretical Foundations

1. **Formal statistical framework**: Develop rigorous hypothesis testing procedures
2. **Power analysis**: Determine minimum sample sizes for reliable detection
3. **Multivariate analysis**: Model joint distribution of indicators
4. **Causal inference**: Integrate causal discovery methods

### 7.2 Empirical Validation

1. **Large-scale studies**: Validate on diverse real-world datasets
2. **Cross-domain validation**: Test generalization across research areas
3. **Ground truth collection**: Build labeled dataset of biased/unbiased evaluations
4. **Threshold calibration**: Empirically determine optimal thresholds

### 7.3 Methodological Extensions

1. **Robust estimators**: Use methods resistant to outliers and violations of assumptions
2. **Non-linear detection**: Incorporate methods for non-linear relationships
3. **Time series methods**: Add trend detection and forecasting
4. **Bayesian framework**: Incorporate prior knowledge and uncertainty quantification

---

## 8. Conclusion

The CBD framework provides a **preliminary, heuristic approach** to detecting circular reasoning bias in AI evaluation. While based on established statistical methods, the framework has significant limitations:

1. **Mathematical foundations**: Indicators have known statistical properties, but specific combinations and thresholds lack rigorous justification
2. **Empirical validation**: Limited validation on real-world data; effectiveness varies by domain
3. **Causal inference**: Framework is correlational; cannot establish causation
4. **Threshold calibration**: Current thresholds are heuristic and not validated

**Recommendation**: Use CBD as a **screening tool** to identify evaluations warranting further investigation, not as definitive evidence of circular bias. Combine with other evaluation integrity practices (pre-registration, held-out test sets, independent validation) for robust conclusions.

---

## References

1. **Coefficient of Variation**: Abdi, H. (2010). Coefficient of variation. Encyclopedia of Research Design, 1, 169-171.

2. **Pearson Correlation**: Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. Proceedings of the Royal Society of London, 58, 240-242.

3. **Effect Size Guidelines**: Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Hillsdale, NJ: Erlbaum.

4. **Correlation vs. Causation**: Pearl, J. (2009). Causality: Models, Reasoning and Inference (2nd ed.). Cambridge University Press.

5. **Time Series Analysis**: Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control (5th ed.). John Wiley & Sons.

---

**Document Status**: Living document, subject to revision as research progresses.

**Contributions Welcome**: We encourage community feedback and contributions to improve the mathematical rigor and empirical validation of the CBD framework.
