"""Unit tests for the Markdown chunker (no models or network needed)."""

from pathlib import Path

from src.chunking import chunk_file, to_url

BODY = "Some body text that is long enough to pass the minimum chunk size of sixty characters."


def make_docs(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "docs" / "en"
    for name, content in files.items():
        f = root / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    return root


def test_to_url():
    assert to_url(Path("modes/train.md")) == "https://docs.ultralytics.com/modes/train/"
    assert to_url(Path("guides/index.md")) == "https://docs.ultralytics.com/guides/"
    assert to_url(Path("index.md")) == "https://docs.ultralytics.com/"


def test_heading_path_is_prepended(tmp_path):
    root = make_docs(tmp_path, {"a.md": f"# Title\n\n## Section\n\n{BODY}\n"})
    chunks = chunk_file(root / "a.md", root)
    assert chunks[0].heading == "Title > Section"
    assert chunks[0].text.startswith("Title > Section")
    assert chunks[0].url == "https://docs.ultralytics.com/a/"


def test_front_matter_is_removed(tmp_path):
    root = make_docs(tmp_path, {"a.md": f"---\ndescription: secret meta\n---\n# Title\n\n{BODY}\n"})
    assert not any("secret meta" in c.text for c in chunk_file(root / "a.md", root))


def test_code_block_is_not_split_on_blank_lines(tmp_path):
    code = "```python\nx = 1\n\ny = 2\n```"
    root = make_docs(tmp_path, {"a.md": f"# T\n\n{BODY}\n\n{code}\n"})
    assert any("x = 1\n\ny = 2" in c.text for c in chunk_file(root / "a.md", root))


def test_large_table_pieces_repeat_the_header(tmp_path):
    header = "| Argument | Type | Default | Description |\n| --- | --- | --- | --- |"
    rows = "\n".join(f"| `arg{i}` | `int` | `1` | Description text number {i} of the argument. |" for i in range(40))
    root = make_docs(tmp_path, {"a.md": f"# T\n\n## Args\n\n{header}\n{rows}\n"})
    chunks = chunk_file(root / "a.md", root)
    assert len(chunks) > 1
    assert all("| Argument | Type | Default | Description |" in c.text for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert all(f"`arg{i}`" in joined for i in range(40))  # no row is lost


def test_macro_include_is_expanded(tmp_path):
    macro = (
        "| Argument | Type | Default | Description |\n| --- | --- | --- | --- |\n"
        "| `conf` | `float` | `0.25` | Minimum confidence threshold for detections. |"
    )
    root = make_docs(tmp_path, {"usage/cfg.md": '# Cfg\n\n## Predict\n\n{% include "macros/args.md" %}\n'})
    (root.parent / "macros").mkdir()
    (root.parent / "macros" / "args.md").write_text(macro, encoding="utf-8")
    chunks = chunk_file(root / "usage" / "cfg.md", root)
    assert any("`conf`" in c.text for c in chunks)
    assert not any("include" in c.text for c in chunks)


def test_table_padding_is_collapsed(tmp_path):
    padded = (
        "| Argument        | Type  | Description                                  |\n"
        "| --------------- | ----- | -------------------------------------------- |\n"
        "| `conf`          | float | Minimum confidence threshold for detections. |"
    )
    root = make_docs(tmp_path, {"a.md": f"# T\n\n{padded}\n"})
    text = chunk_file(root / "a.md", root)[0].text
    assert "  " not in text.split("\n\n", 1)[1]
