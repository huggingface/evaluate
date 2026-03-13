---
title: WER
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
  Word error rate (WER) is a common metric of the performance of an automatic speech recognition system.
  
  The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one). The WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level. The WER is a valuable tool for comparing different systems as well as for evaluating improvements within one system. This kind of measurement, however, provides no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort.
  
  This problem is solved by first aligning the recognized word sequence with the reference (spoken) word sequence using dynamic string alignment. Examination of this issue is seen through a theory called the power law that states the correlation between perplexity and word error rate.
  
  Word error rate can then be computed as:
  
  WER = (S + D + I) / N = (S + D + I) / (S + D + C)
  
  where
  
  S is the number of substitutions,
  D is the number of deletions,
  I is the number of insertions,
  C is the number of correct words,
  N is the number of words in the reference (N=S+D+C).
  
  This value indicates the average number of errors per reference word. The lower the value, the better the
  performance of the ASR system with a WER of 0 being a perfect score.
---

# Metric Card for WER

## Metric description
Word error rate (WER) is a common metric of the performance of an automatic speech recognition (ASR) system. 

The general difficulty of measuring the performance of ASR systems lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one). The WER is derived from the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), working at the word level.

This problem is solved by first aligning the recognized word sequence with the reference (spoken) word sequence using dynamic string alignment. Examination of this issue is seen through a theory called the power law that states the correlation between [perplexity](https://huggingface.co/metrics/perplexity) and word error rate (see [this article](https://www.cs.cmu.edu/~roni/papers/eval-metrics-bntuw-9802.pdf) for further information).

Word error rate can then be computed as:

`WER = (S + D + I) / N = (S + D + I) / (S + D + C)`

where

`S` is the number of substitutions,

`D` is the number of deletions,

`I` is the number of insertions,

`C` is the number of correct words,

`N` is the number of words in the reference (`N=S+D+C`).


## How to use 

The metric takes two inputs: references (a list of references for each speech input) and predictions (a list of transcriptions to score). You can also set `normalize=True` to obtain a normalized WER value.


```python
from evaluate import load
# Standard WER calculation
wer = load("wer")
wer_score = wer.compute(predictions=predictions, references=references)
# Normalized WER calculation
normalized_wer_score = wer.compute(predictions=predictions, references=references, normalize=True)

```
## Output values

This metric outputs a float representing the word error rate.

```
print(wer_score)
0.5
```

This value indicates the average number of errors per reference word. 

The **lower** the value, the **better** the performance of the ASR system, with a WER of 0 being a perfect score.

When using the default settings, WER's output is not always a number between 0 and 1, in particular when there is a high number of insertions (see [Examples](#Examples) below).

When using `normalize=True`, the WER is calculated as `(S + D + I) / (S + D + I + C)`, which ensures the output always falls within the range of 0-1 (or 0-100%).

### Values from popular papers

This metric is highly dependent on the content and quality of the dataset, and therefore users can expect very different values for the same model but on different datasets.

For example, datasets such as [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) report a WER in the 1.8-3.3 range, whereas ASR models evaluated on [Timit](https://huggingface.co/datasets/timit_asr) report a WER in the 8.3-20.4 range. 
See the leaderboards for [LibriSpeech](https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean) and [Timit](https://paperswithcode.com/sota/speech-recognition-on-timit) for the most recent values.

## Examples 

Perfect match between prediction and reference:

```python
from evaluate import load
wer = load("wer")
predictions = ["hello world", "good night moon"]
references = ["hello world", "good night moon"]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
0.0
```

Partial match between prediction and reference:

```python
from evaluate import load
wer = load("wer")
predictions = ["this is the prediction", "there is an other sample"]
references = ["this is the reference", "there is another one"]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
0.5
```

No match between prediction and reference:

```python
from evaluate import load
wer = load("wer")
predictions = ["hello world", "good night moon"]
references = ["hi everyone", "have a great day"]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
1.0
```

WER above 1 due to insertion errors:

```python
from evaluate import load
wer = load("wer")
predictions = ["hello wonderful world and all the people in it"]
references = ["hello world"]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
3.5
# With normalization
normalized_wer_score = wer.compute(predictions=predictions, references=references, normalize=True)
print(normalized_wer_score)
0.7777777777777778
```

## Limitations and bias

WER is a valuable tool for comparing different systems as well as for evaluating improvements within one system. This kind of measurement, however, provides no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort. 

The raw WER can exceed 1.0 when there are many insertion errors. To address this, you can use the `normalize=True` parameter to calculate a normalized WER where the number of errors is divided by the sum of the number of edit operations (`I` + `S` + `D`) and `C` (the number of correct words), which results in WER values that fall within the range of 0–1 (or 0–100%).

## Citation

```bibtex
@inproceedings{woodard1982,
author = {Woodard, J.P. and Nelson, J.T.,
year = {1982},
journal = {Workshop on standardisation for speech I/O technology, Naval Air Development Center, Warminster, PA},
title = {An information theoretic measure of speech recognition performance}
}
```

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

- [Word Error Rate -- Wikipedia](https://en.wikipedia.org/wiki/Word_error_rate)
- [Hugging Face Tasks -- Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)
