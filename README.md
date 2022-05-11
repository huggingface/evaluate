🤗 Evaluate is a library that aims to make evaluating and comparing machine learning models and reporting their performance easier and more standardized. 
```suggestion
🤗 Evaluate is a library that aims to make evaluating and comparing models and reporting their performance easier and more standardized. 


It currently contains:

- **implementations of dozens of popular metrics**: the existing metrics cover a variety of tasks spanning from NLP to Computer Vision, and include dataset-specific metrics for datasets such as [GLUE](https://huggingface.co/datasets/glue) and [SQuAD](https://huggingface.co/datasets/squad). With a simple command like `bleu_score = load_metric("bleu")`, get any of these metrics ready to use for evaluating a ML model in any framework (Numpy/Pandas/PyTorch/TensorFlow/JAX).
- **an easy way of pushing to the 🤗 Hub**: you can create new metrics and push them to a dedicated Space in the 🤗 Hub with `evaluate-cli create [metric name]`, which allows you to see easily compare different metrics and their outputs for the same sets of references and predictions. 

[🎓 **Documentation**](https://huggingface.co/docs/metrics/)

[🔎 **Find a metric in the Hub**](https://huggingface.co/metrics) [🌟 **Add a new dataset to the Hub**](https://github.com/huggingface/rvaluate/blob/master/ADD_NEW_DATASET.md)

🤗 Evaluate also has lots of useful features like:

- **Type checking**: the input types are checked to make sure that you are using the right input formats for each metric
- **Metric cards**: each metrics comes with a card that describes the values, limitations and their ranges, as well as providing examples of their usage and usefulness.
- **Community metrics:** Metrics live on the Hugging Face Hub and you can easily add your own metrics for your project or to collaborate with others.



# Installation

## With pip

🤗 Evaluate can be installed from PyPi and has to be installed in a virtual environment (venv or conda for instance)

```bash
pip install evaluate
```

## With conda

🤗 Evaluate can be installed using conda as follows:


```bash
conda install -c huggingface -c conda-forge evaluate
```

For more details on installation, check the installation page in the documentation: https://huggingface.co/docs/evaluate/installation

## Installation to use with PyTorch/TensorFlow/pandas

If you plan to use 🤗 Evaluate with PyTorch (1.0+), TensorFlow (2.2+) or pandas, you should also install PyTorch, TensorFlow or pandas.

For more details on using the library with NumPy, pandas, PyTorch or TensorFlow, check the quick start page in the documentation: https://huggingface.co/docs/evaluate/quickstart

# Usage

🤗 Evaluate's main methods are:

- `evaluate.list_metrics()` to list the available metrics
- `evaluate.load_metric(metric_name, **kwargs)` to instantiate a metric
- `results = metric.compute(*kwargs)` to compute a metric on a set of predictions and/or references

# Adding a new metric

First install the necessary dependancies to create a new metric with the following command:
```
pip install evaluate[template]
```
Then you can get started with the following command which will create a new folder for your metric and display the necessary steps:
```batch
evaluate-cli create "Awesome Metric"
```
For detailed documentation see ...
TODO : make step-by-step guide similar to the one for [datasets](https://huggingface.co/docs/datasets/share.html).


# Disclaimers

TODO: edit the one below, or write up a new one (+ something about reproducibility?)

Similar to TensorFlow Datasets, 🤗 Datasets is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use them. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/huggingface/datasets/issues/new). Thanks for your contribution to the ML community!

## BibTeX

TODO

```bibtex

```
