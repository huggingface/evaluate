import argparse
import os
import subprocess
from pathlib import Path

from cookiecutter.main import cookiecutter
from huggingface_hub import HfApi, Repository, create_repo


INSTRUCTIONS = """\
A new repository for your module "{module_name}" of type "{module_type}" has been created at {output_dir} and pushed to the Hugging Face Hub: {repo_url}.

Here are the next steps:
- implement the module logic in {module_slug}/{module_slug}.py
- document your module in {module_slug}/README.md
- add test cases for your module in {module_slug}/tests.py
- if your module has any dependencies update them in {module_slug}/requirements.txt

You can test your module's widget locally by running:

```
python {output_dir}/{module_slug}/app.py
```

When you are happy with your changes you can push your changes with the following commands to the Hugging Face Hub:

```
cd {output_dir}/{module_slug}
git add .
git commit -m "Updating module"
git push
```

You should then see the update widget on the Hugging Face Hub: {repo_url}
And you can load your module in Python with the following code:

```
from evaluate import load
module = load("{namespace}/{module_slug}")
```
"""


def main():
    parser = argparse.ArgumentParser("HuggingFace Evaluate CLI tool", usage="evaluate-cli <command> [<args>]")
    subparsers = parser.add_subparsers()
    parser_create = subparsers.add_parser("create", help="Create new evaluation module.")
    parser_create.add_argument(
        "module_name", type=str, help='Pretty name of new evaluation module, e.g. "Recall" or "Exact Match".'
    )
    parser_create.add_argument(
        "--module_type",
        default="metric",
        type=str,
        help="Type of module, has to be one of [metric|comparison|measurement].",
    )
    parser_create.add_argument(
        "--dataset_name", default="", type=str, help="Name of dataset if evaluation module is dataset specific."
    )
    parser_create.add_argument("--module_description", type=str, help="Short description of evaluation module.")
    parser_create.add_argument("--output_dir", default=Path.cwd(), type=str, help="Path to output directory.")
    parser_create.add_argument(
        "--organization", default=None, type=str, help="Organization on the Hub to push evaluation module to."
    )
    parser_create.add_argument("--private", action="store_true", help="Sets evaluation module repository to private.")
    args = vars(parser.parse_args())

    if args["module_type"] not in ["metric", "comparison", "measurement"]:
        raise ValueError("The module_type needs to be one of metric, comparison, or measurement")

    if "-" in args["module_name"]:
        raise ValueError("Hyphens ('-') are not allowed in module names.")

    output_dir = Path(args["output_dir"])
    organization = args["organization"]
    module_slug = args["module_name"].lower().replace(" ", "_")

    if organization is None:
        hfapi = HfApi()
        namespace = hfapi.whoami()["name"]
    else:
        namespace = organization
    args["namespace"] = namespace
    repo_url = f"https://huggingface.co/spaces/{namespace}/{module_slug}"

    create_repo(namespace + "/" + module_slug, repo_type="space", space_sdk="gradio", private=args["private"])
    subprocess.run(
        f"git clone {repo_url}".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=output_dir,
        env=os.environ.copy(),
    )

    repo = Repository(
        local_dir=output_dir / module_slug,
    )

    cookiecutter(
        "https://github.com/huggingface/evaluate/",
        directory="templates",
        no_input=True,
        extra_context=args,
        output_dir=output_dir,
        overwrite_if_exists=True,
    )

    repo.git_add()
    repo.git_commit("add module default template")
    repo.git_push()

    print(
        INSTRUCTIONS.format(
            module_name=args["module_name"],
            module_type=args["module_type"],
            module_slug=module_slug,
            namespace=namespace,
            repo_url=repo_url,
            output_dir=output_dir,
        )
    )


if __name__ == "__main__":
    main()
