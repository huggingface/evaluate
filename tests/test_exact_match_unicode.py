# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pathlib import Path
import evaluate


class ExactMatchUnicodePunctuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metric_path = Path(__file__).resolve().parent.parent / "metrics" / "exact_match"
        cls.exact_match = evaluate.load(str(metric_path))

    def test_ascii_punctuation(self):
        result = self.exact_match.compute(
            predictions=["Cairo,"],
            references=["Cairo"],
            ignore_punctuation=True,
        )
        self.assertEqual(result["exact_match"], 1.0)

    def test_unicode_arabic_punctuation(self):
        """Test that Arabic comma and punctuation are stripped (Issue #836)."""
        result = self.exact_match.compute(
            predictions=["القاهرة،"],
            references=["القاهرة"],
            ignore_punctuation=True,
        )
        self.assertEqual(result["exact_match"], 1.0)

    def test_unicode_cjk_punctuation(self):
        """Test that fullwidth CJK punctuation marks are stripped."""
        result = self.exact_match.compute(
            predictions=["你好，世界！"],
            references=["你好世界"],
            ignore_punctuation=True,
        )
        self.assertEqual(result["exact_match"], 1.0)

    def test_unicode_quotes_and_brackets(self):
        """Test that unicode guillemets and quotation marks are stripped."""
        result = self.exact_match.compute(
            predictions=["«Paris»", "“Rome”", "‘Tokyo’"],
            references=["Paris", "Rome", "Tokyo"],
            ignore_punctuation=True,
        )
        self.assertEqual(result["exact_match"], 1.0)

    def test_unicode_inverted_marks(self):
        """Test that inverted exclamation and question marks are stripped."""
        result = self.exact_match.compute(
            predictions=["¡Hola!", "¿Cómo estás?"],
            references=["Hola", "Cómo estás"],
            ignore_punctuation=True,
        )
        self.assertEqual(result["exact_match"], 1.0)

    def test_ignore_punctuation_false(self):
        """Test that punctuation is preserved when ignore_punctuation=False."""
        result = self.exact_match.compute(
            predictions=["القاهرة،"],
            references=["القاهرة"],
            ignore_punctuation=False,
        )
        self.assertEqual(result["exact_match"], 0.0)
