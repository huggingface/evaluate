# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computes the RL Reliability Metrics."""

import datasets
import numpy as np
from rl_reliability_metrics.evaluation import eval_metrics
from rl_reliability_metrics.metrics import metrics_offline, metrics_online

import evaluate


logger = evaluate.logging.get_logger(__name__)

DEFAULT_EVAL_POINTS = [
    50000,
    150000,
    250000,
    350000,
    450000,
    550000,
    650000,
    750000,
    850000,
    950000,
    1050000,
    1150000,
    1250000,
    1350000,
    1450000,
    1550000,
    1650000,
    1750000,
    1850000,
    1950000,
]

N_RUNS_RECOMMENDED = 10

_CITATION = """\
@conference{rl_reliability_metrics,
  title = {Measuring the Reliability of Reinforcement Learning Algorithms},
  author = {Stephanie CY Chan, Sam Fishman, John Canny, Anoop Korattikara, and Sergio Guadarrama},
  booktitle = {International Conference on Learning Representations, Addis Ababa, Ethiopia},
  year = 2020,
}
"""

_DESCRIPTION = """\
Computes the RL reliability metrics from a set of experiments. There is an `"online"` and `"offline"` configuration for evaluation.
"""


_KWARGS_DESCRIPTION = """
Computes the RL reliability metrics from a set of experiments. There is an `"online"` and `"offline"` configuration for evaluation.
Args:
    timestamps: list of timestep lists/arrays that serve as index.
    rewards: list of reward lists/arrays of each experiment.
Returns:
    dictionary: a set of reliability metrics
Examples:
    >>> import numpy as np
    >>> rl_reliability = evaluate.load("rl_reliability", "online")
    >>> results = rl_reliability.compute(
    ...     timesteps=[np.linspace(0, 2000000, 1000)],
    ...     rewards=[np.linspace(0, 100, 1000)]
    ...     )
    >>> print(results["LowerCVaROnRaw"].round(4))
    [0.0258]
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RLReliability(evaluate.EvaluationModule):
    """Computes the RL Reliability Metrics."""

    def _info(self):
        if self.config_name not in ["online", "offline"]:
            raise KeyError("""You should supply a configuration name selected in '["online", "offline"]'""")

        return evaluate.EvaluationModuleInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "timesteps": datasets.Sequence(datasets.Value("int64")),
                    "rewards": datasets.Sequence(datasets.Value("float")),
                }
            ),
            homepage="https://github.com/google-research/rl-reliability-metrics",
        )

    def _compute(
        self,
        timesteps,
        rewards,
        baseline="default",
        freq_thresh=0.01,
        window_size=100000,
        window_size_trimmed=99000,
        alpha=0.05,
        eval_points=None,
    ):
        if len(timesteps) < N_RUNS_RECOMMENDED:
            logger.warning(
                f"For robust statistics it is recommended to use at least {N_RUNS_RECOMMENDED} runs whereas you provided {len(timesteps)}."
            )

        curves = []
        for timestep, reward in zip(timesteps, rewards):
            curves.append(np.stack([timestep, reward]))

        if self.config_name == "online":
            if baseline == "default":
                baseline = "curve_range"
            if eval_points is None:
                eval_points = DEFAULT_EVAL_POINTS

            metrics = [
                metrics_online.HighFreqEnergyWithinRuns(thresh=freq_thresh),
                metrics_online.IqrWithinRuns(
                    window_size=window_size_trimmed, eval_points=eval_points, baseline=baseline
                ),
                metrics_online.IqrAcrossRuns(
                    lowpass_thresh=freq_thresh, eval_points=eval_points, window_size=window_size, baseline=baseline
                ),
                metrics_online.LowerCVaROnDiffs(baseline=baseline),
                metrics_online.LowerCVaROnDrawdown(baseline=baseline),
                metrics_online.LowerCVaROnAcross(
                    lowpass_thresh=freq_thresh, eval_points=eval_points, window_size=window_size, baseline=baseline
                ),
                metrics_online.LowerCVaROnRaw(alpha=alpha, baseline=baseline),
                metrics_online.MadAcrossRuns(
                    lowpass_thresh=freq_thresh, eval_points=eval_points, window_size=window_size, baseline=baseline
                ),
                metrics_online.MadWithinRuns(
                    eval_points=eval_points, window_size=window_size_trimmed, baseline=baseline
                ),
                metrics_online.MaxDrawdown(),
                metrics_online.StddevAcrossRuns(
                    lowpass_thresh=freq_thresh, eval_points=eval_points, window_size=window_size, baseline=baseline
                ),
                metrics_online.StddevWithinRuns(
                    eval_points=eval_points, window_size=window_size_trimmed, baseline=baseline
                ),
                metrics_online.UpperCVaROnAcross(
                    alpha=alpha,
                    lowpass_thresh=freq_thresh,
                    eval_points=eval_points,
                    window_size=window_size,
                    baseline=baseline,
                ),
                metrics_online.UpperCVaROnDiffs(alpha=alpha, baseline=baseline),
                metrics_online.UpperCVaROnDrawdown(alpha=alpha, baseline=baseline),
                metrics_online.UpperCVaROnRaw(alpha=alpha, baseline=baseline),
                metrics_online.MedianPerfDuringTraining(window_size=window_size, eval_points=eval_points),
            ]
        else:
            if baseline == "default":
                baseline = "median_perf"

            metrics = [
                metrics_offline.MadAcrossRollouts(baseline=baseline),
                metrics_offline.IqrAcrossRollouts(baseline=baseline),
                metrics_offline.StddevAcrossRollouts(baseline=baseline),
                metrics_offline.LowerCVaRAcrossRollouts(alpha=alpha, baseline=baseline),
                metrics_offline.UpperCVaRAcrossRollouts(alpha=alpha, baseline=baseline),
                metrics_offline.MedianPerfAcrossRollouts(baseline=None),
            ]

        evaluator = eval_metrics.Evaluator(metrics=metrics)
        result = evaluator.compute_metrics(curves)
        return result
