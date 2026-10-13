import evaluate


def test_perplexity_gpt2():
    perplexity = evaluate.load("./metrics/perplexity", module_type="metric")

    result = perplexity.compute(
        predictions=["Hello world."],
        model_id="gpt2",
    )

    assert "mean_perplexity" in result
    assert len(result["perplexities"]) == 1


def test_perplexity_long_input():
    perplexity = evaluate.load("./metrics/perplexity", module_type="metric")

    result = perplexity.compute(
        predictions=["Hello world. " * 2000],
        model_id="gpt2",
        add_start_token=False,
    )

    assert "mean_perplexity" in result


