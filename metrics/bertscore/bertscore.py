# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERTScore metric. """

import functools
from contextlib import contextmanager

import bert_score
import datasets
from packaging import version

import evaluate


@contextmanager
def filter_logging_context():
    def filter_log(record):
        return False if "This IS expected if you are initializing" in record.msg else True

    logger = datasets.utils.logging.get_logger("transformers.modeling_utils")
    logger.addFilter(filter_log)
    try:
        yield
    finally:
        logger.removeFilter(filter_log)


_CITATION = """\
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
"""

_DESCRIPTION = """\
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
sentences by cosine similarity.
It has been shown to correlate with human judgment on sentence-level and system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language
generation tasks.

See the project's README at https://github.com/Tiiiger/bert_score#readme for more information.
"""

_KWARGS_DESCRIPTION = """
BERTScore Metrics with the hashcode from a source against one or more references.

Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.
    lang (str): Language of the sentences; required (e.g. 'en').
    model_type (str): Bert specification, default using the suggested
        model for the target language; has to specify at least one of
        `model_type` or `lang`.
    num_layers (int): The layer of representation to use,
        default using the number of layers tuned on WMT16 correlation data.
    verbose (bool): Turn on intermediate status update.
    idf (bool or dict): Use idf weighting; can also be a precomputed idf_dict.
    device (str): On which the contextual embedding model will be allocated on.
        If this argument is None, the model lives on cuda:0 if cuda is available.
    nthreads (int): Number of threads.
    batch_size (int): Bert score processing batch size,
        at least one of `model_type` or `lang`. `lang` needs to be
        specified when `rescale_with_baseline` is True.
    rescale_with_baseline (bool): Rescale bertscore with pre-computed baseline.
    baseline_path (str): Customized baseline file.
    use_fast_tokenizer (bool): `use_fast` parameter passed to HF tokenizer. New in version 0.3.10.

Returns:
    precision: Precision.
    recall: Recall.
    f1: F1 score.
    hashcode: Hashcode of the library.

Examples:

    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> bertscore = evaluate.load("bertscore")
    >>> results = bertscore.compute(predictions=predictions, references=references, lang="en")
    >>> print([round(v, 2) for v in results["f1"]])
    [1.0, 1.0]
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BERTScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/Tiiiger/bert_score",
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/Tiiiger/bert_score"],
            reference_urls=[
                "https://github.com/Tiiiger/bert_score",
                "https://arxiv.org/abs/1904.09675",
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        lang=None,
        model_type=None,
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
    ):

        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        if idf:
            idf_sents = [r for ref in references for r in ref]
        else:
            idf_sents = None

        get_hash = bert_score.utils.get_hash
        scorer = bert_score.BERTScorer

        if version.parse(bert_score.__version__) >= version.parse("0.3.10"):
            get_hash = functools.partial(get_hash, use_fast_tokenizer=use_fast_tokenizer)
            scorer = functools.partial(scorer, use_fast_tokenizer=use_fast_tokenizer)
        elif use_fast_tokenizer:
            raise ImportWarning(
                "To use a fast tokenizer, the module `bert-score>=0.3.10` is required, and the current version of "
                "`bert-score` doesn't match this condition.\n"
                'You can install it with `pip install "bert-score>=0.3.10"`.'
            )

        if model_type is None:
            if lang is None:
                raise ValueError(
                    "Either 'lang' (e.g. 'en') or 'model_type' (e.g. 'microsoft/deberta-xlarge-mnli')"
                    " must be specified"
                )
            model_type = bert_score.utils.lang2model[lang.lower()]

        if num_layers is None:
            num_layers = bert_score.utils.model2layers[model_type]

        hashcode = get_hash(
            model=model_type,
            num_layers=num_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            use_custom_baseline=baseline_path is not None,
        )

        with filter_logging_context():
            if not hasattr(self, "cached_bertscorer") or self.cached_bertscorer.hash != hashcode:
                self.cached_bertscorer = scorer(
                    model_type=model_type,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    nthreads=nthreads,
                    all_layers=all_layers,
                    idf=idf,
                    idf_sents=idf_sents,
                    device=device,
                    lang=lang,
                    rescale_with_baseline=rescale_with_baseline,
                    baseline_path=baseline_path,
                )

        (P, R, F) = self.cached_bertscorer.score(
            cands=predictions,
            refs=references,
            verbose=verbose,
            batch_size=batch_size,
        )
        output_dict = {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F.tolist(),
            "hashcode": hashcode,
        }
        return output_dict
