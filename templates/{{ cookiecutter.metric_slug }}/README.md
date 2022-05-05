---
title: {{ cookiecutter.metric_name }}
datasets:
- {{ cookiecutter.dataset_name }} 
tags:
- metric
sdk: gradio
sdk_version: 2.8.13
app_file: app.py
pinned: false
---

# Metric Card for {{ cookiecutter.metric_name }}

***Metric Card Instructions:*** *Fill out the following subsections. Feel free to take a look at existing metric cards if you'd like examples.*

## Metric Description
*Give a brief overview of this metric, including what task(s) it is usually used for, if any.*

## How to Use
*Give general statement of how to use the metric*

*Provide simplest possible example for using the metric*

### Inputs
*List all input arguments in the format below*
- **input_field** *(type): Definition of input, with explanation if necessary. State any default value(s).*

### Output Values

*Explain what this metric outputs and provide an example of what the metric output looks like. Metrics should return a dictionary with one or multiple key-value pairs, e.g. {"bleu" : 6.02}*

*State the range of possible values that the metric's output can take, as well as what in that range is considered good. For example: "This metric can take on any value between 0 and 100, inclusive. Higher scores are better."*

#### Values from Popular Papers
*Give examples, preferrably with links to leaderboards or publications, to papers that have reported this metric, along with the values they have reported.*

### Examples
*Give code examples of the metric being used. Try to include examples that clear up any potential ambiguity left from the metric description above. If possible, provide a range of examples that show both typical and atypical results, as well as examples where a variety of input parameters are passed.*

## Limitations and Bias
*Note any known limitations or biases that the metric has, with links and references if possible.*

## Citation
*Cite the source where this metric was introduced.*

## Further References
*Add any useful further references.*
