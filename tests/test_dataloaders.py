from datasets import Dataset, load_dataset

from evaluate import evaluator


def test_evaluator_text_classification():
    e = evaluator("text-classification")

    # Test passing in dataset object
    data = Dataset.from_dict(load_dataset("imdb")["test"][:2])
    _, _, dataset = e.prepare_data(data=data, input_column="text", label_column="label", data_split=None)
    assert isinstance(data, Dataset)

    # Test passing in dataset by name with data_split
    _, _, dataset = e.prepare_data(data="imdb", input_column="text", label_column="label", data_split="test[:3]")
    assert isinstance(data, Dataset)

    # Test passing in dataset by name without data_split and inferring the optimal split
    _, _, dataset = e.prepare_data(data="imdb", input_column="text", label_column="label")
    assert isinstance(data, Dataset)


def test_evaluator_question_answering():
    e = evaluator("question-answering")

    # Test passing in dataset object
    data = Dataset.from_dict(load_dataset("squad")["validation"][:2])
    _, _, dataset = e.prepare_data(
        data=data,
        question_column="question",
        context_column="context",
        id_column="id",
        label_column="answers",
        data_split=None,
        squad_v2_format=None,
    )
    assert isinstance(data, Dataset)

    # Test passing in dataset by name with data_split
    _, _, dataset = e.prepare_data(
        data="squad",
        data_split="validation[:3]",
        question_column="question",
        context_column="context",
        id_column="id",
        label_column="answers",
        squad_v2_format=None,
    )
    assert isinstance(data, Dataset)
    # Test passing in dataset by name without data_split and inferring the optimal split
    _, _, dataset = e.prepare_data(
        data="squad",
        data_split=None,
        question_column="question",
        context_column="context",
        id_column="id",
        label_column="answers",
        squad_v2_format=None,
    )


def test_evaluator_token_classification():
    e = evaluator("token-classification")

    # Test passing in dataset object
    data = load_dataset("conll2003", split="validation[:2]")
    _, _, dataset = e.prepare_data(
        data=data,
        input_column="tokens",
        label_column="ner_tags",
        join_by=" ",
        data_split=None,
    )
    assert isinstance(data, Dataset)

    # Test passing in dataset by name with data_split
    _, _, dataset = e.prepare_data(
        data="conll2003",
        data_split="validation[:2]",
        input_column="tokens",
        label_column="ner_tags",
        join_by=" ",
    )
    assert isinstance(data, Dataset)

    # Test passing in dataset by name without data_split and inferring the optimal split
    _, _, dataset = e.prepare_data(
        data="conll2003",
        data_split=None,
        input_column="tokens",
        label_column="ner_tags",
        join_by=" ",
    )
