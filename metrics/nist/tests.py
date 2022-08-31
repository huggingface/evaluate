from _pytest.fixtures import fixture

from nist import NIST

nist = NIST()


@fixture
def hypothesis_sent1():
    return "It is a guide to action which ensures that the military always obeys the commands of the party"


@fixture
def reference_sent1():
    return "It is a guide to action that ensures that the military will forever heed Party commands"


@fixture
def reference_sent2():
    return "It is the guiding principle which guarantees the military forces always being under the command of the Party"


def test_nist_sentence(hypothesis_sent1, reference_sent1, reference_sent2):
    nist_score = nist.compute(predictions=hypothesis_sent1, references=[reference_sent1, reference_sent2])
    assert abs(nist_score - 3.3709935957649324) < 1e-6
