<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# How to contribute to Evaluate

Everyone is welcome to contribute, and we value everybody's contribution. Code
is not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/evaluate/blob/main/CODE_OF_CONDUCT.md).

## You can contribute in so many ways!

There are four ways you can contribute to `evaluate`:
* Fixing outstanding issues with the existing code;
* Implementing new evaluators and metrics;
* Contributing to the examples and documentation;
* Submitting issues related to bugs or desired new features.

Open issues are tracked directly on the repository [here](https://github.com/huggingface/datasets/issues).

If you would like to work on any of the open issues:
* Make sure it is not already assigned to someone else. The assignee (if any) is on the top right column of the Issue page. If it's not taken, self-assign it.
* Work on your self-assigned issue and create a Pull Request!

## Submitting a new issue or feature request

Following these guidelines when submitting an issue or a feature
request will make it easier for us to come back to you quickly and with good
feedback.

### Do you want to implement a new metric?

All evaluation modules, be it metrics, comparisons, or measurements live on the 🤗 Hub in a [Space](https://huggingface.co/docs/hub/spaces) (see for example [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)). Evaluation modules can be either **community** or **canonical**. 

* **Canonical** metrics are well-established metrics which already broadly adopted. 
* **Community** metrics are new or custom metrics. It is simple to add a new community metric to use with `evaluate`. Please see our guide to adding a new evaluation metric [here](https://huggingface.co/docs/evaluate/creating_and_sharing)! 

The only functional difference is that canonical metrics are integrated into the `evaluate` library directly and do not require a namespace when being loaded. 

We encourage contributors to share new evaluation modules they contribute broadly! If they become widely adopted then they will be integrated into the core `evaluate` library as a canonical module.

### Do you want to request a new feature (that is not a metric)?

We would appreciate it if your feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

### Did you find a bug?

Thank you for reporting an issue. If the bug is related to a community metric, please open an issue or pull request directly on the repository of the metric on the Hugging Face Hub. 

If the bug is related to the `evaluate` library and not a community metric, we would really appreciate it if you could **make sure the bug was not already reported** (use the search bar on Github under Issues). If it's not already logged, please open an issue with these details:

* Include your **OS type and version**, the versions of **Python**, **PyTorch** and
  **Tensorflow** when applicable;
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s;
* Provide the *full* traceback if an exception is raised.

## Start contributing! (Pull Requests)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

1. Fork the [repository](https://github.com/huggingface/evaluate) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/evaluate.git
   $ cd evaluate 
   $ git remote add upstream https://github.com/huggingface/evaluate.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   $ pip install -e ".[dev]"
   ```

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this:

   ```bash
   $ pytest tests/<TEST_TO_RUN>.py
   ```
   
   To run a specific test, for example the `test_model_init` test in test_evaluator.py, 

   ```bash
   python -m pytest ./tests/test_evaluator.py::TestQuestionAnsweringEvaluator::test_model_init
   ```

   You can also run the full suite with the following command:

   ```bash
   $ python -m pytest ./tests/ 
   ```

   🤗 Evaluate relies on `black` and `isort` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated in one go with:

   ```bash
   $ make fixup
   ```

   This target is also optimized to only work with files modified by the PR you're working on.

   If you prefer to run the checks one after the other, the following command apply the
   style corrections:

   ```bash
   $ make style
   ```

   🤗 Evaluate also uses `flake8` and a few custom scripts to check for coding mistakes. Quality
   control runs in CI, however you can also run the same checks with:

   ```bash
   $ make quality
   ```

   If you're modifying documents under `docs/source`, make sure to validate that
   they can still be built. This check also runs in CI. To run a local check
   make sure you have installed the documentation builder requirements. First you will need to clone the
   repository containing our tools to build the documentation:
   
   ```bash
   $ pip install git+https://github.com/huggingface/doc-builder
   ```

   Then, make sure you have all the dependencies to be able to build the doc with:
   
   ```bash
   $ pip install ".[docs]"
   ```

   Finally, run the following command from the root of the repository:

   ```bash
   $ doc-builder build evaluate docs/source/ --build_dir ~/tmp/test-build
   ```

   This will build the documentation in the `~/tmp/test-build` folder where you can inspect the generated
   Markdown files with your favorite editor. You won't be able to see the final rendering on the website
   before your PR is merged, we are actively working on adding a tool for this.

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit
   messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied, go to the webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`. These
   are useful to avoid duplicated work, and to differentiate it from PRs ready
   to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.
6. All public methods must have informative docstrings that work nicely with sphinx. 
7. Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
   the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference 
   them by URL.


### Style guide

For documentation strings, 🤗 Evaluate follows the [google style](https://google.github.io/styleguide/pyguide.html).
Check our [documentation writing guide](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)
for more information.

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

### Develop on Windows

On Windows, you need to configure git to transform Windows `CRLF` line endings to Linux `LF` line endings:

`git config core.autocrlf input`

One way one can run the make command on Window is to pass by MSYS2:

1. [Download MSYS2](https://www.msys2.org/), we assume to have it installed in C:\msys64
2. Open the command line C:\msys64\msys2.exe (it should be available from the start menu)
3. Run in the shell: `pacman -Syu` and install make with `pacman -S make`
4. Add `C:\msys64\usr\bin` to your PATH environment variable.

You can now use `make` from any terminal (Powershell, cmd.exe, etc) 🎉

### Syncing forked main with upstream (HuggingFace) main

To avoid pinging the upstream repository which adds reference notes to each upstream PR and sends unnecessary notifications to the developers involved in these PRs,
when syncing the main branch of a forked repository, please, follow these steps:
1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead, merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:
```
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```
