import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from cookiecutter.main import cookiecutter
from huggingface_hub import Repository, create_repo, delete_repo, list_metrics, whoami

from evaluate import __version__
from evaluate.utils.logging import get_logger


logger = get_logger(__name__)

templates_dir = Path(__file__).parent / 'templates'

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


def sanitize_module_name(name: str) -> str:
    if "-" in name:
        logger.error("Hyphens ('-') are not allowed in module names.")
        os.exit(1)
    return name.lower().replace(" ", "_")


def sanitize_organization(name: Optional[str]) -> str:
    return name or whoami()['name']


def create(args: Dict[str, Any]):
    module_slug = sanitize_module_name(args['module_name'])
    namespace = sanitize_organization(args['organization'])

    output_dir: Path = args["output_dir"]
    if output_dir.exists() and not output_dir.is_dir():
        logger.error('Expected path to directory not file: %s', output_dir)
        return
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / module_slug

    args["namespace"] = namespace
    repo_url = f"https://huggingface.co/spaces/{namespace}/{module_slug}"

    logger.debug('render module template in %s', output_dir)
    cookiecutter(
        template=str(templates_dir),
        directory="module",
        no_input=True,
        extra_context=args,
        output_dir=output_dir,
        overwrite_if_exists=True,
    )

    def exec_git(command: str):
        subprocess.run(args=command.split(),
                       stdout=subprocess.DEVNULL,
                       check=True,
                       encoding="utf-8",
                       cwd=local_dir,
                       env=os.environ.copy())

    logger.debug('initialize non-empty git repository at %s', local_dir)
    for command in ['git init', f'git remote add origin {repo_url}']:
        exec_git(command)

    repo = Repository(local_dir=local_dir)
    repo.git_add()
    repo.git_commit("add module default template")

    # By default we create project on organization space, create local git
    # repo, update git repo, make initial commit, and push changes to project.
    # This behavior can be changed with --local option.
    if not args['local']:
        try:
            create_repo(namespace + "/" + module_slug, repo_type="space", space_sdk="gradio", private=args["private"])
        except Exception:
            logger.exception(("Could not create Space for module at "
                              "hf.co/spaces/%s/%s. Make sure this space does not "
                              "exist already."), namespace, module_slug)
            return

        exec_git('git fetch origin')
        exec_git('git branch -u origin/main')
        exec_git('git rebase -X ours origin/main')
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


def delete(args: Dict[str, Any]):
    module_slug = sanitize_module_name(args['module_name'])
    namespace = sanitize_organization(args['organization'])
    try:
        delete_repo(f'{namespace}/{module_slug}', repo_type='space')
    except Exception:
        logger.exception('Could not delete module %s. Does the module exist?',
                         module_slug)


def list_(args: Dict[str, Any]):
    if not (metrics := list_metrics()):
        print('No metrics.')
    for metric in metrics:
        print(metric)


def help_(args: Dict[str, Any]):
    parser.print_help()


def version(args: Dict[str, Any]):
    print(f'evaluate-cli version {__version__}')


def main():
    args = vars(parser.parse_args())
    if (func := args.get('func')) is None:
        parser.print_help()
        return
    func(args)


parser = argparse.ArgumentParser(
    "HuggingFace Evaluate CLI tool", usage="evaluate-cli <command> [<args>]",
    epilog='See https://huggingface.co/docs/evaluate for details.',
)
subparsers = parser.add_subparsers()

parser_create = subparsers.add_parser("create", help="Create new evaluation module.")
parser_create.set_defaults(func=create)
parser_create.add_argument(
    "module_name", type=str, help='Pretty name of new evaluation module, e.g. "Recall" or "Exact Match".'
)
parser_create.add_argument(
    "--organization", default=None, type=str, help="Organization on the Hub to push evaluation module to."
)
parser_create.add_argument(
    "--module_type", default="metric", type=str, choices=("comparison", "measurement", "metric"),
    help="Type of module.",
)
parser_create.add_argument(
    "--dataset_name", default="", type=str, help="Name of dataset if evaluation module is dataset specific."
)
parser_create.add_argument("--module_description", type=str, help="Short description of evaluation module.")
parser_create.add_argument("-o", "--output_dir", default=Path.cwd(), type=Path,
                           metavar='DIR', help="Path to output directory.")
parser_create.add_argument("--local", action="store_true", help="Initialize only local files.")
parser_create.add_argument("--private", action="store_true", help="Sets evaluation module repository to private.")

parser_delete = subparsers.add_parser("delete", help="Delete existing evaluation module.")
parser_delete.set_defaults(func=delete)
parser_delete.add_argument("module_name", type=str, help='Module name to delete.')
parser_delete.add_argument("--organization", default=None, type=str, help="Organization on the Hub.")

parser_list = subparsers.add_parser("list", help="List public metrics.")
parser_list.set_defaults(func=list_)

parser_help = subparsers.add_parser("help", add_help=False, help="Show this help message and exit.")
parser_help.set_defaults(func=help_)

parser_version = subparsers.add_parser("version", add_help=False, help="Show version.")
parser_version.set_defaults(func=version)
