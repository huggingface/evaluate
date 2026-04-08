# Copyright 2025 Responsible AI Labs.
#
# Licensed under the MIT License.
"""RAIL Score metric for responsible AI evaluation of LLM outputs."""

import os
import logging
import time
from typing import Dict, List, Optional

import datasets

import evaluate


logger = logging.getLogger(__name__)


_CITATION = """\
@software{rail_score,
    title = {RAIL Score: Responsible AI Evaluation Framework},
    author = {Responsible AI Labs},
    year = {2025},
    url = {https://responsibleailabs.ai},
    version = {2.4.0}
}
"""

_DESCRIPTION = """\
RAIL Score evaluates LLM outputs across 8 responsible AI dimensions,
providing fine-grained scores (0-10) for each dimension plus an overall
composite score. It helps teams identify fairness issues, safety risks,
reliability gaps, and other responsible AI concerns in model outputs.

Dimensions:
- Fairness: Equitable treatment across demographic groups
- Safety: Prevention of harmful or unsafe content
- Reliability: Factual accuracy and internal consistency
- Transparency: Clear communication of limitations and reasoning
- Privacy: Protection of personal and sensitive information
- Accountability: Traceability and auditability of decisions
- Inclusivity: Accessible and inclusive language
- User Impact: Positive value delivered to the user

Requires a RAIL Score API key (free tier available at https://responsibleailabs.ai).
"""

_KWARGS_DESCRIPTION = """
RAIL Score evaluation of LLM responses across 8 responsible AI dimensions.

Args:
    predictions (list of str): LLM responses to evaluate.
    references (list of str): Input prompts or context for each response (optional).
    api_key (str): RAIL Score API key. Defaults to RAIL_API_KEY environment variable.
        Get a free key at https://responsibleailabs.ai.
    mode (str): "basic" (default) or "deep" for detailed explanations.
    domain (str): Content domain. One of "general", "healthcare", "finance",
        "legal", "education", or "code". Default: "general".
    dimensions (list of str): Subset of dimensions to evaluate. Default: all 8.
    weights (dict): Custom dimension weights for the overall score.
        Values must sum to 100. Default: equal weights (12.5 each).
    include_explanations (bool): Return per-dimension explanations. Default: False.
    include_issues (bool): Return detected issues. Default: False.

Returns:
    overall_score (float): Mean RAIL score across all examples.
    overall_confidence (float): Mean confidence across all examples.
    {dimension} (float): Mean score per dimension.
    {dimension}_confidence (float): Mean confidence per dimension.
    scores (list of float): Per-example overall scores.
    confidences (list of float): Per-example overall confidence values.

Examples:
    >>> import evaluate
    >>> rail_score = evaluate.load("rail_score")
    >>> results = rail_score.compute(
    ...     predictions=["The capital of France is Paris."],
    ...     references=["What is the capital of France?"],
    ... )
    >>> print(round(results["overall_score"], 1))
    8.5
"""

ALL_DIMENSIONS = [
    "fairness",
    "safety",
    "reliability",
    "transparency",
    "privacy",
    "accountability",
    "inclusivity",
    "user_impact",
]


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RAILScore(evaluate.Metric):
    """RAIL Score metric for responsible AI evaluation of LLM outputs."""

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                        "references": datasets.Value("string"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                    }
                ),
            ],
            homepage="https://docs.responsibleailabs.ai",
            codebase_urls=[
                "https://github.com/Responsible-AI-Labs/rail-score-sdk",
                "https://pypi.org/project/rail-score-sdk/",
            ],
            reference_urls=[
                "https://responsibleailabs.ai",
                "https://docs.responsibleailabs.ai",
            ],
            license="MIT",
        )

    def _compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        mode: str = "basic",
        domain: str = "general",
        dimensions: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        include_explanations: bool = False,
        include_issues: bool = False,
    ) -> Dict:
        try:
            from rail_score_sdk import RailScoreClient
        except ImportError:
            raise ImportError(
                "rail-score-sdk is required for RAIL Score. "
                "Install it with: pip install rail-score-sdk"
            )

        key = api_key or os.environ.get("RAIL_API_KEY")
        if not key:
            raise ValueError(
                "RAIL Score API key is required. Either pass api_key= to compute() "
                "or set the RAIL_API_KEY environment variable. "
                "Get a free key at https://responsibleailabs.ai"
            )

        client = RailScoreClient(api_key=key)
        eval_dimensions = dimensions or ALL_DIMENSIONS

        overall_scores: List[float] = []
        overall_confidences: List[float] = []
        dim_scores: Dict[str, List[float]] = {d: [] for d in eval_dimensions}
        dim_confidences: Dict[str, List[float]] = {d: [] for d in eval_dimensions}
        dim_explanations: Optional[Dict[str, List[str]]] = (
            {d: [] for d in eval_dimensions} if include_explanations else None
        )
        all_issues: Optional[List[Dict]] = [] if include_issues else None

        for i, prediction in enumerate(predictions):
            context = references[i] if references is not None else None

            result = self._eval_with_retry(
                client,
                content=prediction,
                mode=mode,
                dimensions=eval_dimensions,
                weights=weights,
                context=context,
                domain=domain,
                include_explanations=include_explanations,
                include_issues=include_issues,
            )

            overall_scores.append(result.rail_score.score)
            overall_confidences.append(result.rail_score.confidence)

            for dim in eval_dimensions:
                if dim in result.dimension_scores:
                    ds = result.dimension_scores[dim]
                    dim_scores[dim].append(ds.score)
                    dim_confidences[dim].append(ds.confidence)
                    if dim_explanations is not None:
                        explanation = getattr(ds, "explanation", None)
                        dim_explanations[dim].append(explanation or "")

            if all_issues is not None and result.issues:
                all_issues.extend(
                    [
                        {
                            "example_index": i,
                            "dimension": issue.dimension,
                            "description": issue.description,
                        }
                        for issue in result.issues
                    ]
                )

            if (i + 1) % 50 == 0:
                logger.info("Evaluated %d/%d examples", i + 1, len(predictions))

        n = len(overall_scores)
        output: Dict = {
            "overall_score": sum(overall_scores) / n if n else 0.0,
            "overall_confidence": sum(overall_confidences) / n if n else 0.0,
            "scores": overall_scores,
            "confidences": overall_confidences,
            "num_examples": n,
        }

        for dim in eval_dimensions:
            scores = dim_scores[dim]
            confs = dim_confidences[dim]
            if scores:
                output[dim] = sum(scores) / len(scores)
                output[f"{dim}_scores"] = scores
                output[f"{dim}_confidence"] = sum(confs) / len(confs)
                output[f"{dim}_confidences"] = confs

        if dim_explanations is not None:
            output["explanations"] = dim_explanations

        if all_issues is not None:
            output["issues"] = all_issues

        return output

    @staticmethod
    def _eval_with_retry(client, max_retries: int = 3, **kwargs):
        """Call client.eval() with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                return client.eval(**kwargs)
            except Exception as e:
                is_rate_limit = "RateLimitError" in type(e).__name__ or "429" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Rate limited, retrying in %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise
