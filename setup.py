# Lint as: python3
""" HuggingFace/Evaluate is an open library for evaluation.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

0. Prerequisites:
   - Dependencies:
     - twine: "pip install twine"
   - Create an account in (and join the 'evaluate' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/

1. Change the version in:
   - __init__.py
   - setup.py

2. Commit these changes: "git commit -m 'Release: VERSION'"

3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for pypi'"
   Push the tag to remote: git push --tags origin main

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   First, delete any "build" directory that may exist from previous builds.

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv/notebook by running:
   pip install huggingface_hub fsspec aiohttp
   pip install -U tqdm
   pip install -i https://testpypi.python.org/pypi evaluate

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Fill release notes in the tag in github once everything is looking hunky-dory.

8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
   Then push the change with a message 'set dev version'
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
    "responses<0.19",
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
    "nltk",  # for NIST and probably others
    "pytest",
    "pytest-datadir",
    "pytest-xdist",
    # optional dependencies
    "tensorflow>=2.3,!=2.6.0,!=2.6.1, <=2.10",
    "torch",
    # metrics dependencies
    "bert_score>=0.3.6",
    "rouge_score>=0.1.2",
    "sacrebleu",
    "sacremoses",
    "scipy",
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
    version="0.3.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
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
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="metrics machine learning evaluate evaluation",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
