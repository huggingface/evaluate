from inspect import getmembers, isfunction, signature, _empty
from sklearn import metrics
from cookiecutter.main import cookiecutter
import subprocess

classification_metrics = ['accuracy_score',
 'auc',
 'average_precision_score',
 'balanced_accuracy_score',
 'brier_score_loss',
 'classification_report',
 'confusion_matrix',
 'dcg_score',
 'det_curve',
 'f1_score',
 'fbeta_score',
 'hamming_loss',
 'hinge_loss',
 'jaccard_score',
 'log_loss',
 'matthews_corrcoef',
 'multilabel_confusion_matrix',
 'ndcg_score',
 'precision_recall_curve',
 'precision_recall_fscore_support',
 'precision_score',
 'recall_score',
 'roc_auc_score',
 'roc_curve',
 'top_k_accuracy_score',
 'zero_one_loss']


classification_features = {
    "y_true": ['datasets.Sequence(datasets.Value("int32"))', 'datasets.Value("int32")'], 
    'y_pred': ['datasets.Sequence(datasets.Value("int32"))', 'datasets.Value("int32")'],
    'y_prob': ['datasets.Value("float")'],
    'y_score': ['datasets.Sequence(datasets.Value("float"))', 'datasets.Value("float")'],
    "y": ['datasets.Sequence(datasets.Value("float"))'],
    "x": ['datasets.Sequence(datasets.Value("float"))'],
    "pred_decision": ['datasets.Sequence(datasets.Value("float"))', 'datasets.Value("float")'],
    "probas_pred": ['datasets.Sequence(datasets.Value("float"))'], 
}


regression_metrics = [
    "explained_variance_score"
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_tweedie_deviance",
    "d2_tweedie_score",
    "mean_pinball_loss",
    "d2_pinball_score",
    "d2_absolute_error_score",
]

regression_features = {
    "y_pred": ['datasets.Value("float")'],
    "y_true": ['datasets.Value("float")']
}

force_args_to_kwargs = {
    "fbeta_score":
        {"beta": 1}
}

scikit_url_template = "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.{metric_name}.html"


def combine_features(reference_features, prediction_features):
    features = []
    for pred_feature in prediction_features:
        for ref_feature in reference_features:
            features.append(f'datasets.Features({{"predictions": {pred_feature}, "references": {ref_feature}}})')
    if len(features)==1:
        return features[0]
    else:
        return "[" + ", ".join(features) + "]"


def run_parser(output_dir="./"):
    for task_list, task_features in zip([classification_metrics, regression_metrics], [classification_features, regression_features]):
        for sk_metric in [func for func in getmembers(metrics, isfunction) if func[0] in task_list]:
            metric_name = sk_metric[0]
            metric_docs = sk_metric[1].__doc__
            metric_docs_url = scikit_url_template.format(metric_name=metric_name)
            metric_signature = signature(sk_metric[1])
            metric_args = [key for key in metric_signature.parameters if metric_signature.parameters[key].default is _empty]
            metric_kwargs = dict([(key, metric_signature.parameters[key].default) for key in metric_signature.parameters if metric_signature.parameters[key].default is not _empty])

            if metric_name in force_args_to_kwargs:
                for arg_to_move in force_args_to_kwargs[metric_name]:
                    metric_args.remove(arg_to_move)
                    metric_kwargs[arg_to_move] = force_args_to_kwargs[metric_name][arg_to_move]

            metric_features = combine_features(task_features[metric_args[0]], task_features[metric_args[1]])

            args = {
                "module_name": metric_name,
                "module_class_name": "".join([m.capitalize() for m in metric_name.split("_")]),
                "namespace": "sklearn",
                "label_name": metric_args[0],
                "preds_name": metric_args[1],
                "features": metric_features,
                "kwargs": ", ".join([f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in metric_kwargs.items()]),
                "kwargs_input": ", ".join([f"{k}={k}" for k in metric_kwargs.keys()]),
                "docs_url": metric_docs_url,
                "docstring": metric_docs,
                "docstring_first_line": metric_docs.split("\n")[0].strip()
            }

            cookiecutter(
                "https://github.com/huggingface/evaluate/",
                directory="integrations/scikit-learn/template",
                no_input=True,
                extra_context=args,
                output_dir=output_dir,
                overwrite_if_exists=True,
            )

    subprocess.Popen(f"black --line-length 119 --target-version py36 {str(output_dir)}".split())
    subprocess.Popen(f"isort {str(output_dir)}".split())