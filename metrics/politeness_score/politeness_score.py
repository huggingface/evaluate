import evaluate

_DESCRIPTION = """
Politeness Score: Assigns a politeness rating (0 = rude, 1 = neutral, 2 = polite) to each text.
Use for evaluating LLM outputs for tone and user experience.
"""

_CITATION = ""

class PolitenessScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description="List of output strings from LLM.",
            features=["predictions"]
        )

    def _compute(self, predictions):
        polite_words = ["please", "thank you", "could you", "would you", "appreciate"]
        rude_words = ["idiot", "stupid", "hate", "shut up", "dumb"]

        results = []
        for text in predictions:
            t = text.lower()
            if any(w in t for w in rude_words):
                results.append(0)
            elif any(w in t for w in polite_words):
                results.append(2)
            else:
                results.append(1)
        return {"politeness_score": results}
