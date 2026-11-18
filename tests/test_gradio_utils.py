from unittest.mock import patch

import pytest

from evaluate.utils.gradio import parse_readme


@pytest.mark.parametrize("front_matter", ["", "---\ntitle: example\n---\n"])
def test_parse_readme_decodes_utf8_with_a_non_utf8_default(tmp_path, front_matter):
    body = "# M\u00e9trique \u8bc4\u4f30\n"
    readme = tmp_path / "README.md"
    readme.write_text(front_matter + body, encoding="utf-8")

    def open_with_non_utf8_default(*args, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return open(*args, **kwargs)

    with patch("evaluate.utils.gradio.open", open_with_non_utf8_default, create=True):
        assert parse_readme(readme) == body


def test_parse_readme_preserves_missing_file_message(tmp_path):
    assert parse_readme(tmp_path / "README.md") == "No README.md found."
