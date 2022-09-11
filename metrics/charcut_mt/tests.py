from charcut_mt import Charcut
from pytest import fixture


charcut = Charcut()


@fixture
def example_saudis():
    return {
        "hyp": "this week the saudis denied information published in the new york times",
        "ref": "saudi arabia denied this week information published in the american new york times",
        "charcut_mt": 0.20915032679738563,
    }


@fixture
def example_estimate():
    return {
        "hyp": "this is in fact an estimate",
        "ref": "this is actually an estimate",
        "charcut_mt": 0.16363636363636364,
    }


@fixture
def example_corpus():
    return {
        "hyp": [
            "this week the saudis denied information published in the new york times",
            "this is in fact an estimate",
        ],
        "ref": [
            "saudi arabia denied this week information published in the american new york times",
            "this is actually an estimate",
        ],
        "charcut_mt": 0.1971153846153846,
    }


def test_example_saudis(example_saudis):
    hyp = example_saudis["hyp"]
    ref = example_saudis["ref"]
    result = charcut.compute(predictions=[hyp], references=[ref])

    assert abs(result["charcut_mt"] - example_saudis["charcut_mt"]) < 0.00000001


def test_example_estimate(example_estimate):
    hyp = example_estimate["hyp"]
    ref = example_estimate["ref"]
    result = charcut.compute(predictions=[hyp], references=[ref])

    assert abs(result["charcut_mt"] - example_estimate["charcut_mt"]) < 0.00000001


def test_corpus(example_corpus):
    hyps = example_corpus["hyp"]
    refs = example_corpus["ref"]
    result = charcut.compute(predictions=hyps, references=refs)

    assert abs(result["charcut_mt"] - example_corpus["charcut_mt"]) < 0.00000001
