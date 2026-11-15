import evaluate


def test_local_metric_reload_after_hash_literal_changes(tmp_path):
    script = tmp_path / "hashtag_count.py"
    source = """import evaluate
from datasets import Features, Value

class HashtagCount(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Count a hashtag", citation="",
            features=Features({"predictions": Value("string")}))

    def _compute(self, predictions):
        return {"count": sum(text.count("HASHTAG") for text in predictions)}
"""
    texts = ["#positive", "#positive", "#negative"]
    script.write_text(source.replace("HASHTAG", "#positive"), encoding="utf-8")
    first = evaluate.load(str(script), keep_in_memory=True)
    assert first.compute(predictions=texts) == {"count": 2}

    script.write_text(source.replace("HASHTAG", "#negative"), encoding="utf-8")
    second = evaluate.load(str(script), keep_in_memory=True)
    assert second.compute(predictions=texts) == {"count": 1}
