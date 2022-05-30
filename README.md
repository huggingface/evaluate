<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/evaluate/media/resolve/main/evaluate-banner.png" width="400"/>
    <br>
<p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/evaluate">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/evaluate/main">
    </a>
    <a href="https://github.com/huggingface/evaluate/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/evaluate.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/evaluate/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/evaluate/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/evaluate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/evaluate.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

🤗 Evaluate is a library that makes evaluating and comparing models and reporting their performance easier and more standardized. 

It currently contains:

- **implementations of dozens of popular metrics**: the existing metrics cover a variety of tasks spanning from NLP to Computer Vision, and include dataset-specific metrics for datasets. With a simple command like `accuracy = load("accuracy")`, get any of these metrics ready to use for evaluating a ML model in any framework (Numpy/Pandas/PyTorch/TensorFlow/JAX).
- **includes comparisons and measurements**: comparisons are used to measure the difference between models and measurements are tools to evaluate datasets.
- **an easy way of adding new evaluation modules to the 🤗 Hub**: you can create new evaluation modules and push them to a dedicated Space in the 🤗 Hub with `evaluate-cli create [metric name]`, which allows you to see easily compare different metrics and their outputs for the same sets of references and predictions.

[🎓 **Documentation**](https://huggingface.co/docs/evaluate/)

🔎 **Find a [metric](https://huggingface.co/evaluate-metric), [comparison](https://huggingface.co/evaluate-comparison), [measurement](https://huggingface.co/evaluate-metric) on the Hub**

[🌟 **Add a new evaluation module**](https://huggingface.co/docs/evaluate/)

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

# Usage

🤗 Evaluate's main methods are:

- `evaluate.list_evaluation_modules()` to list the available metrics, comparisons and measurements
- `evaluate.load(module_name, **kwargs)` to instantiate an evaluation module
- `results = module.compute(*kwargs)` to compute the result of an evaluation module

# Adding a new evaluation module

First install the necessary dependencies to create a new metric with the following command:
```
pip install evaluate[template]
```
Then you can get started with the following command which will create a new folder for your metric and display the necessary steps:
```batch
evaluate-cli create "Awesome Metric"
```
For detailed documentation see ...
TODO : make step-by-step guide similar to the one for [datasets](https://huggingface.co/docs/datasets/share.html).

## Credits

Thanks to @marella for letting us use the `evaluate` namespace on PyPi previously used by his [library](https://github.com/marella/evaluate).
