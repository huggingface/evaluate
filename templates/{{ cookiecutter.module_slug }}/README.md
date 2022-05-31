---
title: {{ cookiecutter.module_name }}
datasets:
- {{ cookiecutter.dataset_name }} 
tags:
- evaluate
- {{ cookiecutter.module_type }}
description: TODO: add a description here
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
---

# {{ cookiecutter.module_type|capitalize }} Card for {{ cookiecutter.module_name }}

***Module Card Instructions:*** *Fill out the following subsections. Feel free to take a look at existing {{ cookiecutter.module_type }} cards if you'd like examples.*

## {{ cookiecutter.module_type|capitalize }} Description
*Give a brief overview of this {{ cookiecutter.module_type }}, including what task(s) it is usually used for, if any.*

## How to Use
*Give general statement of how to use the {{ cookiecutter.module_type }}*

*Provide simplest possible example for using the {{ cookiecutter.module_type }}*

### Inputs
*List all input arguments in the format below*
- **input_field** *(type): Definition of input, with explanation if necessary. State any default value(s).*

### Output Values

*Explain what this {{ cookiecutter.module_type }} outputs and provide an example of what the {{ cookiecutter.module_type }} output looks like. Modules should return a dictionary with one or multiple key-value pairs, e.g. {"bleu" : 6.02}*

*State the range of possible values that the {{ cookiecutter.module_type }}'s output can take, as well as what in that range is considered good. For example: "This {{ cookiecutter.module_type }} can take on any value between 0 and 100, inclusive. Higher scores are better."*

#### Values from Popular Papers
*Give examples, preferrably with links to leaderboards or publications, to papers that have reported this {{ cookiecutter.module_type }}, along with the values they have reported.*

### Examples
*Give code examples of the {{ cookiecutter.module_type }} being used. Try to include examples that clear up any potential ambiguity left from the {{ cookiecutter.module_type }} description above. If possible, provide a range of examples that show both typical and atypical results, as well as examples where a variety of input parameters are passed.*

## Limitations and Bias
*Note any known limitations or biases that the {{ cookiecutter.module_type }} has, with links and references if possible.*

## Citation
*Cite the source where this {{ cookiecutter.module_type }} was introduced.*

## Further References
*Add any useful further references.*
