import os
import tempfile
import unittest

from datasets import DownloadConfig

import evaluate
from evaluate.loading import evaluation_module_factory


class LocalDownloadConfigTest(unittest.TestCase):
    def test_local_script_keeps_download_config_for_extra_modules(self):
        # A local metric can import an external helper. The caller's DownloadConfig
        # has to reach that download; otherwise local_files_only is ignored and the
        # loader tries the network.
        script_dir = tempfile.mkdtemp()
        cache_dir = tempfile.mkdtemp()
        modules = evaluate.loading.init_dynamic_modules(
            name="test_datasets_modules_" + os.path.basename(script_dir),
            hf_modules_cache=tempfile.mkdtemp(),
        )
        script_path = os.path.join(script_dir, "local_with_external.py")
        with open(script_path, "w") as handle:
            handle.write(
                "from .helper import value  # From: https://example.invalid/helper.py\n"
                "import evaluate\n"
                "from evaluate import EvaluationModuleInfo\n"
                "from datasets import Features, Value\n"
                "class LocalWithExternal(evaluate.EvaluationModule):\n"
                "    def _info(self):\n"
                "        return EvaluationModuleInfo(features=Features({'predictions': Value('int'), 'references': Value('int')}))\n"
                "    def _compute(self, predictions, references):\n"
                "        return {'score': value}\n"
            )
        offline_config = DownloadConfig(cache_dir=cache_dir, local_files_only=True)
        with self.assertRaises(FileNotFoundError) as ctx:
            evaluation_module_factory(
                script_path,
                download_config=offline_config,
                dynamic_modules_path=modules,
            )
        self.assertIn("local_files_only", str(ctx.exception))
