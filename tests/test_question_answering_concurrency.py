"""Run against the installed upstream package with pytest."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from datasets import Dataset

from evaluate.evaluator.question_answering import QuestionAnsweringEvaluator


class Pipeline:
    task = "question-answering"

    def __call__(self, question, context, *, handle_impossible_answer):
        return [{"answer": "" if handle_impossible_answer else "Paris", "score": 0.9}]


class Metric:
    def __init__(self, v2):
        self.name = "squad_v2" if v2 else "squad"

    def compute(self, *, predictions, references):
        assert references[0]["id"] == predictions[0]["id"] == "example"
        expected = references[0]["answers"]["text"] or [""]
        return {"exact_match": int(predictions[0]["prediction_text"] in expected)}


def evaluate(runner, v2):
    return runner.compute(
        model_or_pipeline=Pipeline(),
        data=Dataset.from_list(
            [
                {
                    "id": "example",
                    "question": "What is the capital?",
                    "context": "Paris",
                    "answers": {"text": [] if v2 else ["Paris"], "answer_start": [] if v2 else [0]},
                }
            ]
        ),
        metric=Metric(v2),
        squad_v2_format=v2,
        device=-1,
    )


def test_sequential():
    runner = QuestionAnsweringEvaluator()
    for v2 in (True, False, True):
        assert evaluate(runner, v2)["exact_match"] == 1


@pytest.mark.parametrize("same_instance", [False, True], ids=["separate", "same"])
def test_concurrent(monkeypatch, same_instance):
    v2_at_pipeline = Event()
    v1_finished = Event()
    v2_runner = QuestionAnsweringEvaluator()
    v1_runner = v2_runner if same_instance else QuestionAnsweringEvaluator()
    call_pipeline = QuestionAnsweringEvaluator.call_pipeline

    def interleave(self, pipeline, *args, **kwargs):
        if not v2_at_pipeline.is_set():
            v2_at_pipeline.set()
            assert v1_finished.wait(10), "v1 did not finish"
        return call_pipeline(self, pipeline, *args, **kwargs)

    monkeypatch.setattr(QuestionAnsweringEvaluator, "call_pipeline", interleave)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(evaluate, v2_runner, True)
        try:
            assert v2_at_pipeline.wait(10), "v2 did not reach pipeline invocation"
            assert evaluate(v1_runner, False)["exact_match"] == 1
        finally:
            v1_finished.set()
        assert result.result()["exact_match"] == 1, "v2 lost no-answer handling"
