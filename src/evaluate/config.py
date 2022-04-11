import importlib
import os
import platform
from pathlib import Path

from packaging import version

from .utils.logging import get_logger


logger = get_logger(__name__)


# Metrics
S3_METRICS_BUCKET_PREFIX = "https://s3.amazonaws.com/datasets.huggingface.co/datasets/metrics"
CLOUDFRONT_METRICS_DISTRIB_PREFIX = "https://cdn-datasets.huggingface.co/datasets/metric"
REPO_METRICS_URL = "https://raw.githubusercontent.com/huggingface/evaluate/{revision}/metrics/{path}/{name}"

# Hub
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HUB_DATASETS_URL = HF_ENDPOINT + "/datasets/{path}/resolve/{revision}/{name}"
HUB_DEFAULT_VERSION = "main"

PY_VERSION = version.parse(platform.python_version())

if PY_VERSION < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

# General environment variables accepted values for booleans
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


# Imports
PANDAS_VERSION = version.parse(importlib_metadata.version("pandas"))
PYARROW_VERSION = version.parse(importlib_metadata.version("pyarrow"))

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib_metadata.version("torch"))
            logger.info(f"PyTorch version {TORCH_VERSION} available.")
        except importlib_metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib_metadata.version(package))
            except importlib_metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(f"TensorFlow found but with version {TF_VERSION}. `datasets` requires version 2 minimum.")
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")


JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib_metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib_metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")


# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))

DEFAULT_HF_EVALUATE_CACHE = os.path.join(HF_CACHE_HOME, "evaluate")
HF_EVALUATE_CACHE = Path(os.getenv("HF_EVALUATE_CACHE", DEFAULT_HF_EVALUATE_CACHE))

DEFAULT_HF_METRICS_CACHE = os.path.join(HF_CACHE_HOME, "metrics")
HF_METRICS_CACHE = Path(os.getenv("HF_METRICS_CACHE", DEFAULT_HF_METRICS_CACHE))

DEFAULT_HF_MODULES_CACHE = os.path.join(HF_CACHE_HOME, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE))

DOWNLOADED_DATASETS_DIR = "downloads"
DEFAULT_DOWNLOADED_EVALUATE_PATH = os.path.join(HF_EVALUATE_CACHE, DOWNLOADED_DATASETS_DIR)
DOWNLOADED_EVALUATE_PATH = Path(os.getenv("HF_DATASETS_DOWNLOADED_EVALUATE_PATH", DEFAULT_DOWNLOADED_EVALUATE_PATH))

EXTRACTED_EVALUATE_DIR = "extracted"
DEFAULT_EXTRACTED_EVALUATE_PATH = os.path.join(DEFAULT_DOWNLOADED_EVALUATE_PATH, EXTRACTED_EVALUATE_DIR)
EXTRACTED_EVALUATE_PATH = Path(os.getenv("HF_DATASETS_EXTRACTED_EVALUATE_PATH", DEFAULT_EXTRACTED_EVALUATE_PATH))

# Download count for the website
HF_UPDATE_DOWNLOAD_COUNTS = (
    os.environ.get("HF_UPDATE_DOWNLOAD_COUNTS", "AUTO").upper() in ENV_VARS_TRUE_AND_AUTO_VALUES
)

# Offline mode
HF_EVALUATE_OFFLINE = os.environ.get("HF_EVALUATE_OFFLINE", "AUTO").upper() in ENV_VARS_TRUE_VALUES


# File names
LICENSE_FILENAME = "LICENSE"
METRIC_INFO_FILENAME = "metric_info.json"
DATASETDICT_JSON_FILENAME = "dataset_dict.json"

MODULE_NAME_FOR_DYNAMIC_MODULES = "evaluate_modules"
