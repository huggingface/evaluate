from pathlib import Path

import pytest

import evaluate


@pytest.fixture
def word_count():
    pytest.importorskip("sklearn")
    path = Path(__file__).resolve().parents[1] / "measurements" / "word_count"
    return evaluate.load(str(path), module_type="measurement", keep_in_memory=True)


@pytest.mark.parametrize(
    ("max_vocab", "total", "unique"),
    [(None, 6, 3), (2, 5, 2)],
)
def test_word_count(word_count, max_vocab, total, unique):
    result = word_count.compute(data=["Café, café! a BEER.", "Beer café tea."], max_vocab=max_vocab)
    assert result == {"total_word_count": total, "unique_words": unique}


def test_word_count_rejects_empty_vocabulary(word_count):
    with pytest.raises(ValueError, match="empty vocabulary"):
        word_count.compute(data=["a !", "I _"])
