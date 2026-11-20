import unittest

from evaluate.loading import convert_github_url


class ConvertGithubUrlTest(unittest.TestCase):
    def test_repo_root_uses_master_archive(self):
        url, sub = convert_github_url("https://github.com/huggingface/evaluate")
        self.assertEqual(url, "https://github.com/huggingface/evaluate/archive/master.zip")
        self.assertEqual(sub, "evaluate-master")

    def test_trailing_slash_does_not_crash(self):
        url, sub = convert_github_url("https://github.com/huggingface/evaluate/")
        self.assertEqual(url, "https://github.com/huggingface/evaluate/archive/master.zip")
        self.assertEqual(sub, "evaluate-master")

    def test_tree_url_keeps_branch_out_of_the_zip_name(self):
        url, sub = convert_github_url("https://github.com/huggingface/evaluate/tree/main")
        self.assertEqual(url, "https://github.com/huggingface/evaluate/archive/main.zip")
        self.assertEqual(sub, "evaluate-main")

    def test_blob_url_points_at_the_raw_file(self):
        url, sub = convert_github_url("https://github.com/huggingface/evaluate/blob/main/setup.py")
        self.assertEqual(url, "https://github.com/huggingface/evaluate/raw/main/setup.py")
        self.assertIsNone(sub)

    def test_tree_subdirectory_is_inside_the_archive(self):
        url, sub = convert_github_url("https://github.com/huggingface/evaluate/tree/main/metrics")
        self.assertEqual(url, "https://github.com/huggingface/evaluate/archive/main.zip")
        self.assertEqual(sub, "evaluate-main/metrics")
