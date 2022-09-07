from _pytest.fixtures import fixture
from nist_mt import Nist_mt


nist = Nist_mt()


@fixture
def hypothesis_sent():
    return "It is a guide to action which ensures that the military always obeys the commands of the party"


@fixture
def reference_sent1():
    return "It is a guide to action that ensures that the military will forever heed Party commands"


@fixture
def reference_sent2():
    return (
        "It is the guiding principle which guarantees the military forces always being under the command of the Party"
    )


@fixture
def reference_sent3():
    return (
        "It is the practical guide for the army always to heed the directions of the party"
    )


def test_nist_sentence(hypothesis_sent, reference_sent1, reference_sent2, reference_sent3):
    nist_score = nist.compute(predictions=[hypothesis_sent],
                              references=[[reference_sent1, reference_sent2, reference_sent3]])
    assert abs(nist_score["nist_mt"] - 3.3709935957649324) < 1e-6
