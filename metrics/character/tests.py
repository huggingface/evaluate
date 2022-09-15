from statistics import mean, median, stdev

from character import Character
from pytest import fixture


character = Character()


@fixture
def example_saudis():
    return {
        "hyp": "this week the saudis denied information published in the new york times",
        "ref": "saudi arabia denied this week information published in the american new york times",
        "cer": 0.36619718309859156,
    }


@fixture
def example_estimate():
    return {"hyp": "this is in fact an estimate", "ref": "this is actually an estimate", "cer": 0.25925925925925924}


def test_character_single(example_saudis):
    results = character.compute(predictions=example_saudis["hyp"], references=example_saudis["ref"])

    assert abs(results["cer_score"] - example_saudis["cer"]) < 0.00000001


def test_character_corpus(example_saudis, example_estimate):
    hyps = [example_saudis["hyp"], example_estimate["hyp"]]
    refs = [example_saudis["ref"], example_estimate["ref"]]
    results = character.compute(predictions=hyps, references=refs)
    real_results = [example_saudis["cer"], example_estimate["cer"]]

    assert abs(results["mean"] - mean(real_results)) < 0.00000001
    assert abs(results["median"] - median(real_results)) < 0.00000001
    assert abs(results["std"] - stdev(real_results)) < 0.00000001
    assert abs(results["min"] - min(real_results)) < 0.00000001
    assert abs(results["max"] - max(real_results)) < 0.00000001
