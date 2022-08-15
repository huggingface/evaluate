from datasets import Dataset, load_dataset

from evaluate import evaluator


def test_evaluator_text_classification():
    e = evaluator("text-classification")

    # Test passing in dataset object
    data = Dataset.from_dict(load_dataset("imdb")["test"][:2])
    e.prepare_data(data=data, input_column="text", label_column="label")

    # Test passing in dataset by name with data_split
    data = e.load_data("imdb", data_split="test[:3]")
    e.prepare_data(data=data, input_column="text", label_column="label")

    # Test passing in dataset by name without data_split and inferring the optimal split
    data = e.load_data("imdb")
    e.prepare_data(data=data, input_column="text", label_column="label")


def test_evaluator_question_answering():
    e = evaluator("question-answering")

    # Test passing in dataset object
    data = Dataset.from_dict(load_dataset("squad")["validation"][:2])
    e.prepare_data(
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
    data = e.load_data("squad", data_split="validation[:3]")
    e.prepare_data(
        data=data,
        question_column="question",
        context_column="context",
        id_column="id",
        label_column="answers",
        squad_v2_format=None,
    )
    assert isinstance(data, Dataset)
    # Test passing in dataset by name without data_split and inferring the optimal split
    data = e.load_data("squad")
    e.prepare_data(
        data=data,
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
    e.prepare_data(data=data, input_column="tokens", label_column="ner_tags", join_by=" ")
    assert isinstance(data, Dataset)

    # Test passing in dataset by name with data_split
    data = e.load_data("conll2003", data_split="validation[:2]")
    e.prepare_data(
        data=data,
        input_column="tokens",
        label_column="ner_tags",
        join_by=" ",
    )
    assert isinstance(data, Dataset)

    # Test passing in dataset by name without data_split and inferring the optimal split
    data = e.load_data("conll2003")
    e.prepare_data(
        data=data,
        input_column="tokens",
        label_column="ner_tags",
        join_by=" ",
    )
