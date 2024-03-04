---
title: DER
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
  The primary metric utilized in speaker diarization experiments is the Diarization Error Rate (DER), as defined and employed by NIST in the RT evaluations (NIST Fall Rich Transcription on meetings 2006 Evaluation Plan, 2006). DER measures the fraction of time that is incorrectly attributed to a speaker or non-speech segments. To evaluate DER, the MD-eval-v12.pl script (NIST MD-eval-v21 DER evaluation script, 2006), developed by NIST, is utilized.

The diarization output from the system hypothesis does not necessarily need to identify speakers by name or definitive ID. Thus, the ID tags assigned to speakers in both the hypothesis and reference segmentation do not need to match. This contrasts with non-speech tags, which are identified as unlabeled gaps between two speaker segments and therefore must be identified explicitly.

The evaluation script initially establishes an optimal one-to-one mapping of all speaker label IDs between hypothesis and reference files. This facilitates the scoring of different ID tags between the two files.

---

# Metric Card for der

## Metric description

The main metric for speaker diarization experiments is the Diarization Error Rate (DER), defined by NIST in RT evaluations. DER measures the fraction of time incorrectly attributed to speakers or non-speech segments. Evaluation involves comparing hypothesis and reference files, allowing for flexibility in speaker identification. The evaluation script optimally maps speaker label IDs between the files.


## How to use 

The metric takes two inputs: references (a list of references for each speech input) and predictions (a list of transcriptions to score).

```python
from evaluate import load
der = load("der")
der_score = der.compute(predictions=predictions, references=references)
```

### Inputs

Mandatory inputs: 
- `predictions`:  List of tuples (speaker_id, start_time, end_time) representing the diarization hypothesis.

- `references`: List of tuples (speaker_id, start_time, end_time) representing the ground truth diarization.
## Output values

This metric outputs a float representing the character error rate.

```
print(der_score)
0.3500000000000001
```

The **lower** the der value, the **better** the performance of the Speaker Diarization Task, with a DER of 0 being a perfect score. 


### Values from popular papers
## Examples 



```python
from evaluate import load
der = load("der")
ref = [("A", 0.0, 1.0),
       ("B", 1.0, 1.5),
       ("A", 1.6, 2.1)]

# hypothesis (diarization result from your algorithm)
hyp = [("1", 0.0, 0.8),
       ("2", 0.8, 1.4),
       ("3", 1.5, 1.8),
       ("1", 1.8, 2.0)]
der_score = der.compute(predictions=predictions, references=references)
print(der_score)
0.350
```


## Limitations and bias

The Diarization Error Rate (DER) metric, commonly used in speaker diarization experiments, has its limitations and potential biases. One limitation arises from the fact that DER measures errors in speaker segmentation without considering the actual content or context of the speech. As a result, it may not fully capture the quality of the diarization output, especially in cases where speakers have similar acoustic characteristics or there are overlapping speech segments. 


## Citation


```bibtex
@inproceedings{gallibert2013methodologies,
    author = {Olivier Galibert},
    title = {Methodologies for the evaluation of speaker diarization and automatic speech recognition in the presence of overlapping speech},
    booktitle = {Interspeech},
    year = {2013}}
```

## Further References 

- [Speaker diarisation - wikipedia](https://en.wikipedia.org/wiki/Speaker_diarisation)
