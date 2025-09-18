# Lint as: python3
""" HuggingFace/Evaluate is an open library for evaluation.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

To create the package for pypi.

1. Open a PR and change the version in:
   - __init__.py
   - setup.py
   Then merge the PR once it's approved.

3. Add a tag "vVERSION" (e.g. v0.4.1) in git to mark the release : "git tag vVERSION -m 'Add tag vVERSION for pypi'"
   Push the tag to remote: git push --tags origin main
   Then verify that the 'Python release' CI job runs and succeeds.

4. Fill release notes in the tag in github once everything is looking hunky-dory.

5. Open a PR to change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=0.4.1 -> 0.4.2.dev0).
   Then merge the PR once it's approved.
"""

import os

from setuptools import find_packages, setup


REQUIRED_PKGS = [
    # We need datasets as a backend
    "datasets>=2.0.0",
    # We use numpy>=1.17 to have np.random.Generator (Dataset shuffling)
    "numpy>=1.17",
    # For smart caching dataset processing
    "dill",
    # For performance gains with apache arrow
    "pandas",
    # for downloading datasets over HTTPS
    "requests>=2.19.0",
    # progress bars in download and scripts
    "tqdm>=4.62.1",
    # for fast hashing
    "xxhash",
    # for better multiprocessing
    "multiprocess",
    # to get metadata of optional dependencies such as torch or tensorflow for Python versions that don't have it
    "importlib_metadata;python_version<'3.8'",
    # to save datasets locally or on any filesystem
    # minimum 2021.05.0 to have the AbstractArchiveFileSystem
    "fsspec[http]>=2021.05.0",
    # To get datasets from the Datasets Hub on huggingface.co
    "huggingface-hub>=0.7.0",
    # Utilities from PyPA to e.g., compare versions
    "packaging",
]

TEMPLATE_REQUIRE = [
    # to populate metric template
    "cookiecutter",
    # for the gradio widget
    "gradio>=3.0.0"
]

EVALUATOR_REQUIRE = [
   "transformers",
   # for bootstrap computations in Evaluator
   "scipy>=1.7.1",
]

TESTS_REQUIRE = [
    # test dependencies
    "absl-py",
    "charcut>=1.1.1",  # for charcut_mt
    "cer>=1.2.0",  # for characTER
    "nltk",  # for NIST and probably others
    "pytest",
    "pytest-datadir",
    "pytest-xdist",
    # optional dependencies
    "numpy<2.0.0",  # tensorflow requires numpy < 2
    "tensorflow>=2.3,!=2.6.0,!=2.6.1, <=2.10",
    "torch",
    # metrics dependencies
    "accelerate",  # for frugalscore (calls transformers' Trainer)
    "bert_score>=0.3.6",
    "rouge_score>=0.1.2",
    "sacrebleu",
    "sacremoses",
    "scipy>=1.10.0",
    "seqeval",
    "scikit-learn",
    "jiwer",
    "sentencepiece",  # for bleurt
    "transformers", # for evaluator
    "mauve-text",
    "trectools",
    # to speed up pip backtracking
    "toml>=0.10.1",
    "requests_file>=1.5.1",
    "tldextract>=3.1.0",
    "texttable>=1.6.3",
    "unidecode>=1.3.4",
    "Werkzeug>=1.0.1",
    "six~=1.15.0",
]

QUALITY_REQUIRE = ["black~=22.0", "flake8>=3.8.3", "isort>=5.0.0", "pyyaml>=5.3.1"]


EXTRAS_REQUIRE = {
    "tensorflow": ["tensorflow>=2.2.0,!=2.6.0,!=2.6.1"],
    "tensorflow_gpu": ["tensorflow-gpu>=2.2.0,!=2.6.0,!=2.6.1"],
    "torch": ["torch"],
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "docs": [
        # Might need to add doc-builder and some specific deps in the future
        "s3fs",
    ],
    "template": TEMPLATE_REQUIRE,
    "evaluator": EVALUATOR_REQUIRE
}

setup(
    name="evaluate",
    version="0.4.6",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="HuggingFace community-driven open-source library of evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="leandro@huggingface.co",
    url="https://github.com/huggingface/evaluate",
    download_url="https://github.com/huggingface/evaluate/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": ["evaluate-cli=evaluate.commands.evaluate_cli:main"]},
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="metrics machine learning evaluate evaluation",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
