import os
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets.utils.file_utils import (
    OfflineModeIsEnabled,
    cached_path,
    ftp_get,
    ftp_head,
    http_get,
    http_head,
)


FILE_CONTENT = """\
    Text data.
    Second line of data."""


def test_cached_path_local(text_file):
    # absolute path
    text_file = str(Path(text_file).resolve())
    assert cached_path(text_file) == text_file
    # relative path
    text_file = str(Path(__file__).resolve().relative_to(Path(os.getcwd())))
    assert cached_path(text_file) == text_file


def test_cached_path_missing_local(tmp_path):
    # absolute path
    missing_file = str(tmp_path.resolve() / "__missing_file__.txt")
    with pytest.raises(FileNotFoundError):
        cached_path(missing_file)
    # relative path
    missing_file = "./__missing_file__.txt"
    with pytest.raises(FileNotFoundError):
        cached_path(missing_file)


@patch("datasets.config.HF_DATASETS_OFFLINE", True)
def test_cached_path_offline():
    with pytest.raises(OfflineModeIsEnabled):
        cached_path("https://huggingface.co")


@patch("datasets.config.HF_DATASETS_OFFLINE", True)
def test_http_offline(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "file.html"
    with pytest.raises(OfflineModeIsEnabled):
        http_get("https://huggingface.co", temp_file=filename)
    with pytest.raises(OfflineModeIsEnabled):
        http_head("https://huggingface.co")


@patch("datasets.config.HF_DATASETS_OFFLINE", True)
def test_ftp_offline(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "file.html"
    with pytest.raises(OfflineModeIsEnabled):
        ftp_get("ftp://huggingface.co", temp_file=filename)
    with pytest.raises(OfflineModeIsEnabled):
        ftp_head("ftp://huggingface.co")
