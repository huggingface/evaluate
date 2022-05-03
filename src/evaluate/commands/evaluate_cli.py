import argparse
import os
import subprocess
from pathlib import Path

from cookiecutter.main import cookiecutter
from huggingface_hub import HfApi, Repository, create_repo


INSTRUCTIONS = """\
A new repository for your metric "{metric_name}" has been created at {output_dir} and pushed to the Hugging Face Hub: {repo_url}.

Here are the next steps:
- implement the metric logic in {metric_slug}/{metric_slug}.py
- document your metric in {metric_slug}/README.md
- add test cases for your metric in {metric_slug}/tests.py
- if your metric has any dependencies update them in {metric_slug}/requirements.txt

You can test your metric's widget locally by running:

```
python {output_dir}/{metric_slug}/app.py
```

When you are happy with your changes you can push your changes with the following commands to the Hugging Face Hub:

```
cd {output_dir}/{metric_slug}
git add .
git commit -m "Updating metric"
git push
```

You should then see the update widget on the Hugging Face Hub: {repo_url}
And you can load your metric in Python with the following code:

```
from evaluate import load_metric
metric = load_metric("{namespace}/{metric_slug}")
```
"""


def main():
    parser = argparse.ArgumentParser("HuggingFace Evaluate CLI tool", usage="evaluate-cli <command> [<args>]")
    subparsers = parser.add_subparsers()
    parser_create = subparsers.add_parser("create", help="Create new metric.")
    parser_create.add_argument(
        "metric_name", type=str, help='Pretty name of new metric, e.g. "Recall" or "Exact Match".'
    )
    parser_create.add_argument(
        "--dataset_name", default="", type=str, help="Name of dataset if metric is dataest specific."
    )
    parser_create.add_argument("--metric_description", type=str, help="Short description of metric.")
    parser_create.add_argument("--output_dir", default=Path.cwd(), type=str, help="Path to output directory.")
    parser_create.add_argument(
        "--organization", default=None, type=str, help="Organization on the Hub to push metric to."
    )
    parser_create.add_argument("--private", action="store_true", help="Sets metric repository to private.")
    args = vars(parser.parse_args())

    output_dir = Path(args["output_dir"])
    organization = args["organization"]
    metric_slug = args["metric_name"].lower().replace(" ", "_")

    if organization is None:
        hfapi = HfApi()
        namespace = hfapi.whoami()["name"]
    else:
        namespace = organization
    repo_url = f"https://huggingface.co/spaces/{namespace}/{metric_slug}"

    create_repo(namespace + "/" + metric_slug, repo_type="space", space_sdk="gradio", private=args["private"])
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
        local_dir=output_dir / metric_slug,
    )

    cookiecutter("./templates", no_input=True, extra_context=args, output_dir=output_dir, overwrite_if_exists=True)

    repo.git_add()
    repo.git_commit("add metric default template")
    repo.git_push()

    print(
        INSTRUCTIONS.format(
            metric_name=args["metric_name"],
            metric_slug=metric_slug,
            namespace=namespace,
            repo_url=repo_url,
            output_dir=output_dir,
        )
    )


if __name__ == "__main__":
    main()
