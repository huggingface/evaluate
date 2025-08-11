from pathlib import Path
from huggingface_hub import create_repo, repo_exists, Repository
import tempfile
import subprocess
import os
import shutil
import logging
import re
from urllib.parse import urlparse

import huggingface_hub
from huggingface_hub.utils._headers import _http_user_agent

logger = logging.getLogger(__name__)

GIT_UP_TO_DATE = "On branch main\nYour branch is up to date with 'origin/main'.\
\n\nnothing to commit, working tree clean\n"
GIT_USER = os.getenv("GIT_USER", None)
GIT_EMAIL = os.getenv("GIT_EMAIL", None)

COMMIT_PLACEHOLDER = "{COMMIT_PLACEHOLDER}"

def _http_ci_user_agent(*args, **kwargs):
    ua = _http_user_agent(*args, **kwargs)
    return ua + os.environ.get("CI_HEADERS", "")


def get_git_tag(lib_path, commit_hash):
    # check if commit has a tag, see: https://stackoverflow.com/questions/1474115/how-to-find-the-tag-associated-with-a-given-git-commit
    command = f"git describe --exact-match {commit_hash}"
    output = subprocess.run(command.split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            cwd=lib_path,
            env=os.environ.copy(),
        )
    tag = output.stdout.strip()
    if re.match(r"v\d*\.\d*\.\d*", tag) is not None:
        return tag
    else:
        return None


def copy_recursive(source_base_path, target_base_path):
    """Copy directory recursively and overwrite existing files."""
    for item in source_base_path.iterdir():
        target_path = target_base_path / item.name
        if item.is_dir():
            target_path.mkdir(exist_ok=True)
            copy_recursive(item, target_path)
        else:
            shutil.copy(item, target_path)

def update_evaluate_dependency(requirements_path, commit_hash):
    """Updates the evaluate requirement with the latest commit."""
    with open(requirements_path, "r") as f:
        file_content = f.read()
    file_content = file_content.replace(COMMIT_PLACEHOLDER, commit_hash)
    with open(requirements_path, "w") as f:
        f.write(file_content)

def push_module_to_hub(module_path, type, token, commit_hash, tag=None):
    module_name = module_path.stem
    org = f"evaluate-{type}"
    
    if not repo_exists(org + "/" + module_name, repo_type="space", token=token):
        create_repo(org + "/" + module_name, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)    
    repo_path = Path(tempfile.mkdtemp())

    repo_url = f"https://user:{token}@huggingface.co/spaces/" + org + "/" + module_name
    clean_repo_url = re.sub(r"(https?)://.*@", r"\1://", repo_url)
    
    try:
        subprocess.run(
            f"git clone {repo_url}".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo_path,
            env=os.environ.copy(),
        )
    except OSError:
        # make sure we don't accidentally expose token
        raise OSError(f"Could not clone from '{clean_repo_url}'")

    repo = Repository(local_dir=repo_path / module_name, token=token, git_user=GIT_USER, git_email=GIT_EMAIL)
    
    copy_recursive(module_path, repo_path / module_name)
    update_evaluate_dependency(repo_path / module_name / "requirements.txt", commit_hash)
    
    repo.git_add()
    try:
        repo.git_commit(f"Update Space (evaluate main: {commit_hash[:8]})")
        repo.git_push()
        logger.info(f"Module '{module_name}' pushed to the hub")
    except OSError as error:
        if str(error) == GIT_UP_TO_DATE:
            logger.info(f"Module '{module_name}' is already up to date.")
        else:
            raise error

    if tag is not None:
        repo.add_tag(tag, message="add evaluate tag", remote="origin")
    
    shutil.rmtree(repo_path)


if __name__ == "__main__":
    huggingface_hub.utils._headers._http_user_agent = _http_ci_user_agent
    evaluation_paths = ["metrics", "comparisons", "measurements"]
    evaluation_types = ["metric", "comparison", "measurement"]

    token = os.getenv("HF_TOKEN")
    evaluate_lib_path = Path(os.getenv("EVALUATE_LIB_PATH"))
    commit_hash = os.getenv("GIT_HASH")
    git_tag = get_git_tag(evaluate_lib_path, commit_hash)
    if git_tag is not None:
        logger.info(f"Found tag: {git_tag}.")

    for type, dir in zip(evaluation_types, evaluation_paths):
        if (evaluate_lib_path/dir).exists():
            for module_path in (evaluate_lib_path/dir).iterdir():
                if module_path.is_dir():
                    logger.info(f"Updating: module {module_path.name}.")
                    push_module_to_hub(module_path, type, token, commit_hash, tag=git_tag)
        else:
            logger.warning(f"No folder {str(evaluate_lib_path/dir)} for {type} found.")
