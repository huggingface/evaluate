---
title: CER
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  Character error rate (CER) is a common metric of the performance of an automatic speech recognition system.
  
  CER is similar to Word Error Rate (WER), but operates on character instead of word. Please refer to docs of WER for further information.
  
  Character error rate can be computed as:
  
  CER = (S + D + I) / N = (S + D + I) / (S + D + C)
  
  where
  
  S is the number of substitutions,
  D is the number of deletions,
  I is the number of insertions,
  C is the number of correct characters,
  N is the number of characters in the reference (N=S+D+C).
  
  CER's output is not always a number between 0 and 1, in particular when there is a high number of insertions. This value is often associated to the percentage of characters that were incorrectly predicted. The lower the value, the better the
  performance of the ASR system with a CER of 0 being a perfect score.
---

# Metric Card for CER

## Metric description

Character error rate (CER) is a common metric of the performance of an automatic speech recognition (ASR) system. CER is similar to Word Error Rate (WER), but operates on character instead of word. 

Character error rate can be computed as: 

`CER = (S + D + I) / N = (S + D + I) / (S + D + C)`

where

`S` is the number of substitutions, 

`D` is the number of deletions, 

`I` is the number of insertions, 

`C` is the number of correct characters, 

`N` is the number of characters in the reference (`N=S+D+C`). 


## How to use 

The metric takes two inputs: references (a list of references for each speech input) and predictions (a list of transcriptions to score). You can also set `normalize=True` to obtain a normalized CER value.

```python
from evaluate import load
cer = load("cer")
# Standard CER calculation
cer_score = cer.compute(predictions=predictions, references=references)
# Normalized CER calculation
normalized_cer_score = cer.compute(predictions=predictions, references=references, normalize=True)
```
## Output values

This metric outputs a float representing the character error rate.

```
print(cer_score)
0.34146341463414637
```

The **lower** the CER value, the **better** the performance of the ASR system, with a CER of 0 being a perfect score. 

When using the default settings, CER's output is not always a number between 0 and 1, in particular when there is a high number of insertions (see [Examples](#Examples) below).

When using `normalize=True`, the CER is calculated as `(S + D + I) / (S + D + I + C)`, which ensures the output always falls within the range of 0-1 (or 0-100%).

### Values from popular papers

This metric is highly dependent on the content and quality of the dataset, and therefore users can expect very different values for the same model but on different datasets.

Multilingual datasets such as [Common Voice](https://huggingface.co/datasets/common_voice) report different CERs depending on the language, ranging from 0.02-0.03 for languages such as French and Italian, to 0.05-0.07 for English (see [here](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice/ASR/CTC) for more values).

## Examples 

Perfect match between prediction and reference:

```python
from evaluate import load
cer = load("cer")
predictions = ["hello world", "good night moon"]
references = ["hello world", "good night moon"]
cer_score = cer.compute(predictions=predictions, references=references)
print(cer_score)
0.0
```

Partial match between prediction and reference:

```python
from evaluate import load
cer = load("cer")
predictions = ["this is the prediction", "there is an other sample"]
references = ["this is the reference", "there is another one"]
cer_score = cer.compute(predictions=predictions, references=references)
print(cer_score)
0.34146341463414637
```

No match between prediction and reference:

```python
from evaluate import load
cer = load("cer")
predictions = ["hello"]
references = ["gracias"]
cer_score = cer.compute(predictions=predictions, references=references)
print(cer_score)
1.0
```

CER above 1 due to insertion errors:

```python
from evaluate import load
cer = load("cer")
predictions = ["hello world"]
references = ["hello"]
cer_score = cer.compute(predictions=predictions, references=references)
print(cer_score)
1.2
# With normalization
normalized_cer_score = cer.compute(predictions=predictions, references=references, normalize=True)
print(normalized_cer_score)
0.54545454545454545  # Will always be between 0 and 1
```

## Limitations and bias

CER is useful for comparing different models for tasks such as automatic speech recognition (ASR) and optic character recognition (OCR), especially for multilingual datasets where WER is not suitable given the diversity of languages. However, CER provides no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort.

The raw CER can exceed 1.0 when there are many insertion errors. To address this, you can use the `normalize=True` parameter to calculate a normalized CER where the number of errors is divided by the sum of the number of edit operations (`I` + `S` + `D`) and `C` (the number of correct characters), which results in CER values that fall within the range of 0–1 (or 0–100%).


## Citation


```bibtex
@inproceedings{morris2004,
author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
year = {2004},
month = {01},
pages = {},
title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
```

## Further References 

- [Hugging Face Tasks -- Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)
