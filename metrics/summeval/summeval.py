"""SummEval"""

from typing import Any

import datasets
import evaluate


_CITATION = """\
@article{fabbri2020summeval,
  title={SummEval: Re-evaluating Summarization Evaluation},
  author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal={arXiv preprint arXiv:2007.12626},
  year={2020}
}
"""

_DESCRIPTION = """\
SummEval is an extensible and unified toolkit for summarization model evaluation.
"""

_KWARGS_DESCRIPTION = """
Compute SummEval evaluation metric.

Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.

Examples:
    >>> from evaluate import load
    >>> summeval_metric = load('summeval', 'rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = summeval_metric.compute(predictions=predictions, references=references)
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SummEval(evaluate.Metric):

    _config_names_list = [
        "rouge", 
        "rouge-we", 
        "mover-score", 
        "bert-score", 
        "summa-qa", 
        "blanc", 
        "supert", 
        "meteor", 
        "s3", 
        "data-stats", 
        "cider", 
        "chrf", 
        "bleu", 
        "syntactic", 
    ]

    def _info(self):
        if self.config_name not in self._config_names_list:
            raise KeyError(
                "You should supply a configuration name selected in "
                f'{self._config_names_list}'
            )
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/Yale-LILY/SummEval"],
            reference_urls=[
                "https://github.com/Yale-LILY/SummEval", 
                "https://arxiv.org/pdf/2007.12626"
            ], 
        )
    
    def _compute(self, predictions, references, **kwargs):
        if self.config_name == "rouge":
            return self._compute_rouge(
                predictions, references, **kwargs
            )
        elif self.config_name == "rouge-we":
            return self._compute_rouge_we(
                predictions, references, **kwargs
            )
        elif self.config_name == "mover-score":
            return self._compute_mover_score(
                predictions, references, **kwargs
            )
        elif self.config_name == "bert-score":
            return self._compute_bert_score(
                predictions, references, **kwargs
            )
        elif self.config_name == "summa-qa":
            return self._compute_summa_qa(
                predictions, references, **kwargs
            )
        elif self.config_name == "blanc":
            return self._compute_blanc(
                predictions, references, **kwargs
            )
        elif self.config_name == "supert":
            return self._compute_supert(
                predictions, references, **kwargs
            )
        elif self.config_name == "meteor":
            return self._compute_meteor(
                predictions, references, **kwargs
            )
        elif self.config_name == "s3":
            return self._compute_s3(
                predictions, references, **kwargs
            )
        elif self.config_name == "data-stats":
            return self._compute_data_stats(
                predictions, references, **kwargs
            )
        elif self.config_name == "cider":
            return self._compute_cider(
                predictions, references, **kwargs
            )
        elif self.config_name == "chrf":
            return self._compute_chrf(
                predictions, references, **kwargs
            )
        elif self.config_name == "bleu":
            return self._compute_bleu(
                predictions, references, **kwargs
            )
        elif self.config_name == "syntactic":
            return self._compute_syntactic(
                predictions, references, **kwargs
            )
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                f'{self._config_names_list}'
            )

    def _compute_rouge(self, predictions, references, aggregate: bool = True):
        from summ_eval.rouge_metric import RougeMetric

        metric = RougeMetric()
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)
        
        return score
    
    def _compute_rouge_we(
        self, 
        predictions, 
        references, 
        n_gram: int = 3, 
        n_workers: int = 24, 
        tokenize: bool = True, 
        aggregate: bool = True
    ):
        from summ_eval.rouge_we_metric import RougeWeMetric

        metric = RougeWeMetric(n_gram=n_gram, n_workers=n_workers, tokenize=tokenize)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_mover_score(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        n_gram: int = 1, 
        remove_subwords: bool = True, 
        batch_size: int = 48
    ):
        from summ_eval.mover_score_metric import MoverScoreMetric

        metric = MoverScoreMetric(n_gram=n_gram, remove_subwords=remove_subwords, batch_size=batch_size)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_bert_score(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        lang: str = 'en',
        model_type: str = 'bert-base-uncased',
        num_layers: int = 8,
        verbose: bool = False,
        idf: bool = False,
        nthreads: int = 4,
        batch_size: int = 64,
        rescale_with_baseline: bool = False
    ):
        from summ_eval.bert_score_metric import BertScoreMetric

        metric = BertScoreMetric(
            lang=lang, 
            model_type=model_type, 
            num_layers=num_layers, 
            verbose=verbose, 
            idf=idf, 
            nthreads=nthreads, 
            batch_size=batch_size, 
            rescale_with_baseline=rescale_with_baseline
        )
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_summa_qa(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        batch_size: int = 8, 
        max_seq_len: int = 384, 
        use_gpu: bool = True, 
        tokenize: bool = True
    ):
        from summ_eval.summa_qa_metric import SummaQAMetric

        metric = SummaQAMetric(
            batch_size=batch_size, max_seq_len=max_seq_len, use_gpu=use_gpu, tokenize=tokenize
        )
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_blanc(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        device: str = 'cuda', 
        inference_batch_size: int = 128, 
        finetune_batch_size: int = 24, 
        use_tune: bool = True
    ):
        from summ_eval.blanc_metric import BlancMetric

        metric = BlancMetric(
            device=device, 
            inference_batch_size=inference_batch_size, 
            finetune_batch_size=finetune_batch_size, 
            use_tune=use_tune
        )
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_supert(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        ref_metric: str = 'top15',
        sim_metric: str = 'f1'
    ):
        from summ_eval.supert_metric import SupertMetric

        metric = SupertMetric(ref_metric=ref_metric, sim_metric=sim_metric)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_meteor(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
    ):
        from summ_eval.meteor_metric import MeteorMetric

        metric = MeteorMetric()
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_s3(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        n_workers: int = 24, 
        tokenize: bool = True
    ):
        from summ_eval.s3_metric import S3Metric

        metric = S3Metric(n_workers=n_workers, tokenize=tokenize)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_data_stats(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        n_gram: int = 3, 
        n_workers: int = 24, 
        case: bool = False, 
        tokenize: bool = True
    ):
        from summ_eval.data_stats_metric import DataStatsMetric

        metric = DataStatsMetric(
            n_gram=n_gram, 
            n_workers=n_workers, 
            case=case, 
            tokenize=tokenize
        )
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_cider(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        n_gram: int = 4, 
        sigma: float = 6, 
        tokenize: bool = True
    ):
        from summ_eval.cider_metric import CiderMetric

        metric = CiderMetric(n_gram=n_gram, sigma=sigma, tokenize=tokenize)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_chrf(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        ncorder: int = 6, 
        beta: int = 2, 
        n_workers: int = 24
    ):
        from summ_eval.chrfpp_metric import ChrfppMetric

        metric = ChrfppMetric(ncorder=ncorder, beta=beta, n_workers=n_workers)
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_bleu(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
        sent_smooth_method: str = 'exp', 
        sent_smooth_value: Any = None, 
        sent_use_effective_order: bool = True, 
        smooth_method: str = 'exp', 
        smooth_value: Any = None, 
        force: bool = False, 
        lowercase: bool = False, 
        use_effective_order: bool = False, 
        n_workers: int = 24
    ):
        from summ_eval.bleu_metric import BleuMetric

        metric = BleuMetric(
            sent_smooth_method=sent_smooth_method, 
            sent_smooth_value=sent_smooth_value, 
            sent_use_effective_order=sent_use_effective_order, 
            smooth_method=smooth_method, 
            smooth_value=smooth_value, 
            force=force, 
            lowercase=lowercase, 
            use_effective_order=use_effective_order, 
            n_workers=n_workers, 
        )
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    
    def _compute_syntactic(
        self, 
        predictions, 
        references, 
        aggregate: bool = True, 
    ):
        from summ_eval.syntactic_metric import SyntacticMetric

        metric = SyntacticMetric()
        score = metric.evaluate_batch(predictions, references, aggregate=aggregate)

        return score
    