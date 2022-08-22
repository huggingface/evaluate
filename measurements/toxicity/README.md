---
title: Toxicity
emoji: 🤗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- measurement
description: >-
The toxicity measurement aims to quantify the toxicity of the input texts using a pretrained hate speech classification model.
---

# Measurement Card for Toxicity

## Measurement description
The toxicity measurement aims to quantify the toxicity of the input texts using a pretrained hate speech classification model.

## How to use

The default model used is [roberta-hate-speech-dynabench-r4](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target). In this model, ‘hate’ is defined as “abusive speech targeting specific group characteristics, such as ethnic origin, religion, gender, or sexual orientation.” Definitions used by other classifiers may vary.
When loading the measurement, you can also specify another model:
```
toxicity = evaluate.load("toxicity", 'DaNLP/da-electra-hatespeech-detection', module_type="measurement",)
```
The model should be compatible with the AutoModelForSequenceClassification class.
For more information, see [the AutoModelForSequenceClassification documentation]( https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForSequenceClassification).

Args:
    `predictions` (list of str): prediction/candidate sentences
    `toxic_label` (optional): the toxic label that you want to detect, depending on the labels that the model has been trained on.
        This can be found using the `id2label` function, e.g.:
            ```
            >>> model = AutoModelForSequenceClassification.from_pretrained("DaNLP/da-electra-hatespeech-detection")
            >>> model.config.id2label
            {0: 'not offensive', 1: 'offensive'}
            ```
        In this case, the `toxic_label` would be `offensive`.
    `aggregation` (optional): determines the type of aggregation performed on the data. If set to `None`, the scores for each prediction are returned.
     Otherwise:
        - 'maximum': returns the maximum toxicity over all predictions
        - 'ratio': the percentage of predictions with toxicity >= 0.5.


## Output values

    `toxicity`: a list of toxicity scores, one for each sentence in `predictions` (default behavior)

    `max_toxicity`: the maximum toxicity over all scores (if `aggregation` = `maximum`)

    `toxicity_ratio` : the percentage of predictions with toxicity >= 0.5 (if `aggregation` = `ratio`)


### Values from popular papers


## Examples
    Example 1 (default behavior):
```
>>> toxicity = evaluate.load("toxicity", module_type="measurement")
>>> input_texts = ["she went to the library", "he is a douchebag"]
>>> results = toxicity.compute(predictions=input_texts)
>>> print(results)
{'toxicity': [0.0002070277841994539, 0.856372594833374]}
```
    Example 2 (returns ratio of toxic sentences):
```
>>> toxicity = evaluate.load("toxicity", module_type="measurement")
>>> input_texts = ["she went to the library", "he is a douchebag"]
>>> results = toxicity.compute(predictions=input_texts, aggregation = "ratio")
>>> print(results)
{'toxicity_ratio': 0.5}
```
    Example 3 (returns the maximum toxicity score):
```
>>> toxicity = evaluate.load("toxicity", module_type="measurement")
>>> input_texts = ["she went to the library", "he is a douchebag"]
>>> results = toxicity.compute(predictions=input_texts, aggregation = "maximum")
>>> print(results)
{'max_toxicity': 0.856372594833374}
```
    Example 4 (uses a custom model):
```
>>> toxicity = evaluate.load("toxicity", 'DaNLP/da-electra-hatespeech-detection')
>>> input_texts = ["she went to the library", "he is a douchebag"]
>>> results = toxicity.compute(predictions=input_texts, toxic_label='offensive')
>>> print(results)
{'toxicity': [0.017554517835378647, 0.020320769399404526]}
```



## Citation

```bibtex
@inproceedings{vidgen2021lftw,
  title={Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection},
  author={Bertie Vidgen and Tristan Thrush and Zeerak Waseem and Douwe Kiela},
  booktitle={ACL},
  year={2021}
}
```

```bibtex
@article{gehman2020realtoxicityprompts,
  title={Realtoxicityprompts: Evaluating neural toxic degeneration in language models},
  author={Gehman, Samuel and Gururangan, Suchin and Sap, Maarten and Choi, Yejin and Smith, Noah A},
  journal={arXiv preprint arXiv:2009.11462},
  year={2020}
}

```

## Further References
