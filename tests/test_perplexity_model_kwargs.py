from unittest.mock import patch

from measurements.perplexity.perplexity import Perplexity


def test_perplexity_forwards_pretrained_kwargs():
    metric = Perplexity()
    with patch("measurements.perplexity.perplexity.AutoModelForCausalLM.from_pretrained") as mock_model, patch(
        "measurements.perplexity.perplexity.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch.object(metric, "_compute", wraps=metric._compute) as wrapped:
        # Stop before the forward pass; we only care about the from_pretrained calls.
        mock_model.side_effect = RuntimeError("stop-after-load")
        try:
            metric._compute(
                data=["hello"],
                model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                model_kwargs={"token": "hf_test_token"},
                tokenizer_kwargs={"token": "hf_test_token", "trust_remote_code": True},
            )
        except RuntimeError as exc:
            assert str(exc) == "stop-after-load"

    mock_model.assert_called_once_with("meta-llama/Meta-Llama-3.1-8B-Instruct", token="hf_test_token")
    mock_tokenizer.assert_called_once_with(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token="hf_test_token",
        trust_remote_code=True,
    )
    wrapped.assert_called_once()
