---
title: RL Reliability
datasets:
-  
tags:
- evaluate
- metric
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
---

# Metric Card for RL Reliability

## Metric Description
The RL Reliability Metrics library provides a set of metrics for measuring the reliability of reinforcement learning (RL) algorithms. 

## How to Use

```python
import evaluate
import numpy as np

rl_reliability = evaluate.load("rl_reliability", "online")
results = rl_reliability.compute(
    timesteps=[np.linspace(0, 2000000, 1000)],
    rewards=[np.linspace(0, 100, 1000)]
    )

rl_reliability = evaluate.load("rl_reliability", "offline")
results = rl_reliability.compute(
    timesteps=[np.linspace(0, 2000000, 1000)],
    rewards=[np.linspace(0, 100, 1000)]
    )
```


### Inputs
- **timesteps** *(List[int]): For each run a an list/array with its timesteps.*
- **rewards** *(List[float]): For each run a an list/array with its rewards.*

KWARGS:
- **baseline="default"** *(Union[str, float]) Normalization used for curves. When `"default"` is passed the curves are normalized by their range in the online setting and by the median performance across runs in the offline case. When a float is passed the curves are divided by that value.*
- **eval_points=[50000, 150000, ..., 2000000]** *(List[int]) Statistics will be computed at these points*
- **freq_thresh=0.01** *(float) Frequency threshold for low-pass filtering.*
- **window_size=100000** *(int) Defines a window centered at each eval point.*
- **window_size_trimmed=99000** *(int) To handle shortened curves due to differencing*
- **alpha=0.05** *(float)The "value at risk" (VaR) cutoff point, a float in the range [0,1].*

### Output Values

In `"online"` mode:
- HighFreqEnergyWithinRuns
- IqrWithinRuns
- IqrAcrossRuns
- LowerCVaROnDiffs
- LowerCVaROnDrawdown
- LowerCVaROnAcross
- LowerCVaROnRaw
- MadAcrossRuns
- MadWithinRuns
- MaxDrawdown
- StddevAcrossRuns
- StddevWithinRuns
- UpperCVaROnAcross
- UpperCVaROnDiffs
- UpperCVaROnDrawdown
- UpperCVaROnRaw
- MedianPerfDuringTraining

In `"offline"` mode:
- MadAcrossRollouts
- IqrAcrossRollouts
- StddevAcrossRollouts
- LowerCVaRAcrossRollouts
- UpperCVaRAcrossRollouts
- MedianPerfAcrossRollouts


### Examples
First get the sample data from the repository:

```bash
wget https://storage.googleapis.com/rl-reliability-metrics/data/tf_agents_example_csv_dataset.tgz
tar -xvzf tf_agents_example_csv_dataset.tgz
```

Load the sample data:
```python
dfs = [pd.read_csv(f"./csv_data/sac_humanoid_{i}_train.csv") for i in range(1, 4)]
```

Compute the metrics:
```python
rl_reliability = evaluate.load("rl_reliability", "online")
rl_reliability.compute(timesteps=[df["Metrics/EnvironmentSteps"] for df in dfs],
                       rewards=[df["Metrics/AverageReturn"] for df in dfs])
```

## Limitations and Bias
This implementation of RL reliability metrics does not compute permutation tests to determine whether algorithms are statistically different in their metric values and also does not compute bootstrap confidence intervals on the rankings of the algorithms.

## Citation

```bitex
@conference{rl_reliability_metrics,
  title = {Measuring the Reliability of Reinforcement Learning Algorithms},
  author = {Stephanie CY Chan, Sam Fishman, John Canny, Anoop Korattikara, and Sergio Guadarrama},
  booktitle = {International Conference on Learning Representations, Addis Ababa, Ethiopia},
  year = 2020,
}
```

## Further References
- Homepage: https://github.com/google-research/rl-reliability-metrics
