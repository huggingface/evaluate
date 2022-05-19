import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import __version__


def save(path_or_file, **data):
    """
    Saves results to a JSON file. Also saves system information such as current time, current commit
    hash if inside a repository, and Python system information.

    Args:
        path_or_file (``str``): Path or file to store the file. If only a folder is provided
            the results file will be saved in the format `"result-%Y_%m_%d-%H_%M_%S.json"`.

    Example:
        ```py
        >>> import evaluate
        >>> result = {"bleu", 0.7}
        >>> params = {"model", "gpt-2"}
        >>> evaluate.save("./results/", **result, **params)
        ```
    """
    current_time = datetime.now()

    file_path = _setup_path(path_or_file, current_time)

    data["_timestamp"] = current_time.isoformat()
    data["_git_commit_hash"] = _git_commit_hash()
    data["_evaluate_version"] = __version__
    data["_python_version"] = sys.version
    data["_interpreter_path"] = sys.executable

    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


def _setup_path(path_or_file, current_time):
    path_or_file = Path(path_or_file)
    is_file = len(path_or_file.suffix) > 0
    if is_file:
        folder = path_or_file.parent
        file_name = path_or_file.name
    else:
        folder = path_or_file
        file_name = "result-" + current_time.strftime("%Y_%m_%d-%H_%M_%S") + ".json"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / file_name


def _git_commit_hash():
    res = subprocess.run("git rev-parse --is-inside-work-tree".split(), cwd="./", stdout=subprocess.PIPE)
    if res.stdout.decode().strip() == "true":
        res = subprocess.run("git rev-parse HEAD".split(), cwd="./", stdout=subprocess.PIPE)
        return res.stdout.decode().strip()
    else:
        return None
