# Metric Card for Google BLEU


## Metric Description
The BLEU score has some undesirable properties when used for single sentences, as it was designed to be a corpus measure. The Google BLEU score is designed to limit these undesirable properties when used for single sentences.

To calculate this score, all sub-sequences of 1, 2, 3 or 4 tokens in output and target sequence (n-grams) are recorded. The precision and recall, described below, are then computed.

- **precision:** the ratio of the number of matching n-grams to the number of total n-grams in the generated output sequence
- **recall:** the ratio of the number of matching n-grams to the number of total n-grams in the target (ground truth) sequence

The minimum value of precision and recall is then returned as the score.


## Intended Uses
This metric is generally used to evaluate machine translation models. It is especially used when scores of individual (prediction, reference) sentence pairs are needed, as opposed to when averaging over the (prediction, reference) scores for a whole corpus. That being said, it can also be used when averaging over the scores for a whole corpus.

Because it performs better on individual sentence pairs as compared to BLEU, Google BLEU has also been used in RL experiments.

## How to Use
This metric takes a list of predicted sentences, as well as a list of references.

```python
sentence1 = "the cat sat on the mat"
sentence2 = "the cat ate the mat"
google_bleu = evaluate.load_metric("google_bleu")
result = google_bleu.compute(predictions=[sentence1], references=[[sentence2]])
print(result)
>>> {'google_bleu': 0.3333333333333333}
```

### Inputs
- **predictions** (list of str): list of translations to score.
- **references** (list of list of str): list of lists of references for each translation.
- **tokenizer** : approach used for tokenizing `predictions` and `references`.
The default tokenizer is `tokenizer_13a`, a minimal tokenization approach that is equivalent to `mteval-v13a`, used by WMT. This can be replaced by any function that takes a string as input and returns a list of tokens as output.
- **min_len** (int): The minimum order of n-gram this function should extract. Defaults to 1.
- **max_len** (int): The maximum order of n-gram this function should extract. Defaults to 4.

### Output Values
This metric returns the following in a dict:
- **google_bleu** (float): google_bleu score

The output format is as follows:
```python
{'google_bleu': google_bleu score}
```

This metric can take on values from 0 to 1, inclusive. Higher scores are better, with 0 indicating no matches, and 1 indicating a perfect match.

Note that this score is symmetrical when switching output and target. This means that, given two sentences, `sentence1` and `sentence2`, whatever score is output when `sentence1` is the predicted sentence and  `sencence2` is the reference sentence will be the same as when the sentences are swapped and `sentence2` is the predicted sentence while `sentence1` is the reference sentence. In code, this looks like:

```python
predictions = "the cat sat on the mat"
references = "the cat ate the mat"
google_bleu = evaluate.load_metric("google_bleu")
result_a = google_bleu.compute(predictions=[predictions], references=[[references]])
result_b = google_bleu.compute(predictions=[predictions], references=[[references]])
print(result_a == result_b)
>>> True
```

#### Values from Popular Papers


### Examples
Example with one reference per sample:
```python
>>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', 'he read the book because he was interested in world history']
>>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat'], ['he was interested in world history because he read the book']]
>>> google_bleu = evaluate.load_metric("google_bleu")
>>> results = google_bleu.compute(predictions=predictions, references=references)
>>> print(round(results["google_bleu"], 2))
0.44
```

Example with multiple references for the first sample:
```python
>>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', 'he read the book because he was interested in world history']
>>> references  = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', 'It is a guide to action that ensures that the rubber duck will never heed the cat commands', 'It is the practical guide for the rubber duck army never to heed the directions of the cat'], ['he was interested in world history because he read the book']]
>>> google_bleu = evaluate.load_metric("google_bleu")
>>> results = google_bleu.compute(predictions=predictions, references=references)
>>> print(round(results["google_bleu"], 2))
0.61
```

Example with multiple references for the first sample, and with `min_len` adjusted to `2`, instead of the default `1`, which means that the function extracts n-grams of length `2`:
```python
>>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', 'he read the book because he was interested in world history']
>>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', 'It is a guide to action that ensures that the rubber duck will never heed the cat commands', 'It is the practical guide for the rubber duck army never to heed the directions of the cat'], ['he was interested in world history because he read the book']]
>>> google_bleu = evaluate.load_metric("google_bleu")
>>> results = google_bleu.compute(predictions=predictions, references=references, min_len=2)
>>> print(round(results["google_bleu"], 2))
0.53
```

Example with multiple references for the first sample, with `min_len` adjusted to `2`, instead of the default `1`, and `max_len` adjusted to `6` instead of the default `4`:
```python
>>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', 'he read the book because he was interested in world history']
>>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', 'It is a guide to action that ensures that the rubber duck will never heed the cat commands', 'It is the practical guide for the rubber duck army never to heed the directions of the cat'], ['he was interested in world history because he read the book']]
>>> google_bleu = evaluate.load_metric("google_bleu")
>>> results = google_bleu.compute(predictions=predictions,references=references, min_len=2, max_len=6)
>>> print(round(results["google_bleu"], 2))
0.4
```

## Limitations and Bias

The GoogleBLEU metric does not come with a predefined tokenization function; previous versions simply used `split()` to split the input strings into tokens. Using a tokenizer such as the default one, `tokenizer_13a`, makes results more standardized and reproducible. The BLEU and sacreBLEU metrics also use this default tokenizer.

## Citation
```bibtex
@misc{wu2016googles,
title={Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation},
author={Yonghui Wu and Mike Schuster and Zhifeng Chen and Quoc V. Le and Mohammad Norouzi and Wolfgang Macherey and Maxim Krikun and Yuan Cao and Qin Gao and Klaus Macherey and Jeff Klingner and Apurva Shah and Melvin Johnson and Xiaobing Liu and Łukasz Kaiser and Stephan Gouws and Yoshikiyo Kato and Taku Kudo and Hideto Kazawa and Keith Stevens and George Kurian and Nishant Patil and Wei Wang and Cliff Young and Jason Smith and Jason Riesa and Alex Rudnick and Oriol Vinyals and Greg Corrado and Macduff Hughes and Jeffrey Dean},
year={2016},
eprint={1609.08144},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```
## Further References
- This Hugging Face implementation uses the [nltk.translate.gleu_score implementation](https://www.nltk.org/_modules/nltk/translate/gleu_score.html)
