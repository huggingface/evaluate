# Copyright 2026 The HuggingFace Evaluate Authors.
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
"""Financial LLM Faithfulness metric — evaluates LLM outputs in regulated financial contexts."""

import re
from typing import List, Optional

import datasets

import evaluate


_CITATION = """\
@misc{bajaj2026financialllmfaithfulness,
  title={Financial LLM Faithfulness: A Metric for Evaluating LLM Outputs in Regulated Financial Contexts},
  author={Bajaj, Priyanka},
  year={2026},
  url={https://github.com/huggingface/evaluate}
}
"""

_DESCRIPTION = """\
Financial LLM Faithfulness evaluates the faithfulness and compliance of LLM-generated outputs
in regulated financial environments (FCA, MiFID II, Basel III, SR 11-7).

It measures three orthogonal properties:

1. **Numerical faithfulness** — whether numerical claims (percentages, monetary values, rates,
   basis points) in the prediction are grounded in the reference. Ungrounded numbers are a
   primary source of hallucination risk in financial AI.

2. **Disclaimer presence** — whether the output contains a required risk disclaimer, as mandated
   by FCA COBS 4 and MiFID II Article 24 for investment-related communications.

3. **Compliance risk score** — a composite score (0–1, lower is safer) combining hallucinated
   value rate and missing disclaimer penalty, suitable for use as a CI/CD regression gate.

This metric requires no external model or API — all checks are rule-based and deterministic,
making it suitable for offline evaluation pipelines in compliance-restricted environments.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `str`):
        LLM-generated output strings to evaluate.
    references (`list` of `str`):
        Ground-truth or source context strings used to check numerical faithfulness.
    require_disclaimer (`bool`, *optional*, defaults to `True`):
        Whether to check for the presence of a risk disclaimer in each prediction.
    disclaimer_patterns (`list` of `str`, *optional*):
        Regex patterns used to detect a disclaimer. Defaults to FCA COBS 4 / MiFID II style
        patterns covering "capital at risk", "past performance", "not financial advice", etc.
    currency_symbols (`list` of `str`, *optional*):
        Currency symbols to recognise when extracting monetary values.
        Defaults to `["£", "$", "€", "¥", "CHF"]`.
    tolerance (`float`, *optional*, defaults to `0.01`):
        Fractional tolerance when comparing numerical values between prediction and reference.
        A tolerance of 0.01 allows ±1% difference (e.g. for rounding).

Returns:
    `dict` with the following keys:
    - **numerical_faithfulness** (`list` of `float`): Per-prediction score (0–1). 1.0 means
      all numerical claims in the prediction are grounded in the reference.
    - **disclaimer_present** (`list` of `bool`): Whether a disclaimer was detected per prediction.
    - **hallucinated_values** (`list` of `list` of `str`): Numerical values found in the
      prediction but not supported by the reference, per prediction.
    - **compliance_risk_score** (`list` of `float`): Composite risk score per prediction
      (0–1, lower is safer). Combines hallucination rate and missing disclaimer penalty.
    - **overall_faithfulness** (`float`): Mean numerical faithfulness across all predictions.
    - **disclaimer_rate** (`float`): Fraction of predictions containing a required disclaimer.
    - **mean_compliance_risk** (`float`): Mean compliance risk score across all predictions.

Examples:

    >>> import evaluate
    >>> metric = evaluate.load("financial_llm_faithfulness")
    >>> predictions = [
    ...     "The fund returned 8.5% last year. Past performance is not a reliable indicator of future results.",
    ...     "Invest now — guaranteed 15% annual return with zero risk.",
    ... ]
    >>> references = [
    ...     "The fund returned 8.5% in the previous 12-month period.",
    ...     "The fund has historically returned between 4% and 7% annually.",
    ... ]
    >>> results = metric.compute(predictions=predictions, references=references)
    >>> print(results["numerical_faithfulness"])
    [1.0, 0.0]
    >>> print(results["disclaimer_present"])
    [True, False]
    >>> print(results["compliance_risk_score"])
    [0.0, 1.0]
"""

# Default disclaimer patterns — FCA COBS 4.2 / MiFID II Art. 24 style
_DEFAULT_DISCLAIMER_PATTERNS = [
    r"past performance (?:is )?not (?:a )?(?:reliable )?indicator",
    r"capital (?:is )?at risk",
    r"not financial advice",
    r"not a financial adviser",
    r"investments? (?:can|may) (?:go down|fall|lose value)",
    r"value of (?:your )?investments? (?:can|may) (?:go down|fall)",
    r"you may (?:get back )?less than you invest",
    r"seek independent (?:financial )?advice",
    r"for (?:general )?information(?:al)? purposes? only",
    r"does not constitute (?:financial|investment) advice",
    r"this is not (?:a )?(?:personal )?(?:financial|investment) (?:advice|recommendation)",
]

# Regex patterns for financial numerical values
_PERCENTAGE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_BASIS_POINTS_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:bps?|basis points?)", re.IGNORECASE)
_MONETARY_PATTERN = re.compile(
    r"(?:£|\$|€|¥|CHF)\s*(\d[\d,]*(?:\.\d+)?(?:[KkMmBbTt](?:illion|n)?)?)\b"
)
_MULTIPLIER_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(billion|million|trillion|thousand)\b", re.IGNORECASE
)

_MULTIPLIERS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_SUFFIX_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}


def _normalise_number(value_str: str) -> float:
    """Convert a string like '1.5M' or '2,500' to a float."""
    value_str = value_str.replace(",", "").strip()
    suffix = value_str[-1].lower() if value_str and value_str[-1].lower() in _SUFFIX_MULTIPLIERS else None
    if suffix:
        return float(value_str[:-1]) * _SUFFIX_MULTIPLIERS[suffix]
    return float(value_str)


def _extract_numbers(text: str) -> List[float]:
    """Extract all financial numerical values from text as normalised floats."""
    numbers = []

    # Percentages
    for m in _PERCENTAGE_PATTERN.finditer(text):
        numbers.append(float(m.group(1)))

    # Basis points → convert to float for comparison
    for m in _BASIS_POINTS_PATTERN.finditer(text):
        numbers.append(float(m.group(1)))

    # Monetary values
    for m in _MONETARY_PATTERN.finditer(text):
        try:
            numbers.append(_normalise_number(m.group(1)))
        except ValueError:
            pass

    # Multiplier phrases ("2.5 billion", "500 million")
    for m in _MULTIPLIER_PATTERN.finditer(text):
        try:
            base = float(m.group(1))
            mult = _MULTIPLIERS.get(m.group(2).lower(), 1)
            numbers.append(base * mult)
        except ValueError:
            pass

    return numbers


def _is_grounded(value: float, reference_values: List[float], tolerance: float) -> bool:
    """Check whether a value is within tolerance of any value in the reference set."""
    if not reference_values:
        return False
    for ref_val in reference_values:
        if ref_val == 0:
            if abs(value) <= tolerance:
                return True
        else:
            if abs(value - ref_val) / abs(ref_val) <= tolerance:
                return True
    return False


def _check_disclaimer(text: str, patterns: List[str]) -> bool:
    """Return True if any disclaimer pattern matches anywhere in the text."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class FinancialLlmFaithfulness(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            homepage="https://github.com/huggingface/evaluate",
            codebase_urls=["https://github.com/huggingface/evaluate"],
            reference_urls=[
                "https://www.handbook.fca.org.uk/handbook/COBS/4/",
                "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014L0065",
            ],
        )

    def _compute(
        self,
        predictions: List[str],
        references: List[str],
        require_disclaimer: bool = True,
        disclaimer_patterns: Optional[List[str]] = None,
        currency_symbols: Optional[List[str]] = None,
        tolerance: float = 0.01,
    ):
        if disclaimer_patterns is None:
            disclaimer_patterns = _DEFAULT_DISCLAIMER_PATTERNS

        numerical_faithfulness = []
        disclaimer_present = []
        hallucinated_values = []
        compliance_risk_scores = []

        for prediction, reference in zip(predictions, references):
            # --- Numerical faithfulness ---
            pred_numbers = _extract_numbers(prediction)
            ref_numbers = _extract_numbers(reference)

            hallucinated = []
            if pred_numbers:
                for val in pred_numbers:
                    if not _is_grounded(val, ref_numbers, tolerance):
                        hallucinated.append(str(val))
                faith_score = 1.0 - len(hallucinated) / len(pred_numbers)
            else:
                # No numerical claims — treat as perfectly faithful
                faith_score = 1.0

            hallucinated_values.append(hallucinated)
            numerical_faithfulness.append(round(faith_score, 4))

            # --- Disclaimer check ---
            has_disclaimer = _check_disclaimer(prediction, disclaimer_patterns)
            disclaimer_present.append(has_disclaimer)

            # --- Compliance risk score ---
            hallucination_risk = 1.0 - faith_score
            disclaimer_penalty = 0.3 if (require_disclaimer and not has_disclaimer) else 0.0
            risk = min(1.0, hallucination_risk + disclaimer_penalty)
            compliance_risk_scores.append(round(risk, 4))

        overall_faithfulness = (
            round(sum(numerical_faithfulness) / len(numerical_faithfulness), 4)
            if numerical_faithfulness
            else 0.0
        )
        disclaimer_rate = (
            round(sum(disclaimer_present) / len(disclaimer_present), 4)
            if disclaimer_present
            else 0.0
        )
        mean_compliance_risk = (
            round(sum(compliance_risk_scores) / len(compliance_risk_scores), 4)
            if compliance_risk_scores
            else 0.0
        )

        return {
            "numerical_faithfulness": numerical_faithfulness,
            "disclaimer_present": disclaimer_present,
            "hallucinated_values": hallucinated_values,
            "compliance_risk_score": compliance_risk_scores,
            "overall_faithfulness": overall_faithfulness,
            "disclaimer_rate": disclaimer_rate,
            "mean_compliance_risk": mean_compliance_risk,
        }
