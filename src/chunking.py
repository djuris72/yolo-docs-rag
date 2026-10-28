"""Split Ultralytics Markdown docs into heading-aware chunks."""

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1500
TABLE_CHARS = 800  # tables are cut into small pieces so one argument is not buried among dozens
MIN_CHARS = 60
DOCS_BASE_URL = "https://docs.ultralytics.com"

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
INCLUDE = re.compile(r"\{%\s*include\s+[\"']([^\"']+)[\"']\s*%\}")
IFRAME = re.compile(r"<iframe.*?</iframe>", re.S)
HTML_TAG = re.compile(r"</?(?:p|br|img|strong|div|span)\b[^>]*>")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass
class Chunk:
    id: str
    text: str  # text that gets embedded (heading path + body)
    path: str  # source file, relative to docs/en
    url: str  # public docs URL, used for citations
    heading: str  # heading path


def expand_includes(md: str, docs_dir: Path) -> str:
    """Inline MkDocs macro includes (e.g. the argument tables); unresolved includes are dropped."""

    def repl(m):
        f = docs_dir / m.group(1)
        return f.read_text(encoding="utf-8") if f.exists() else ""

    return INCLUDE.sub(repl, md)


def clean_prose(line: str) -> str:
    line = MD_IMAGE.sub("", line)
    line = MD_LINK.sub(r"\1", line)
    line = HTML_TAG.sub("", line)
    if line.lstrip().startswith("|"):  # table padding only aligns columns and wastes embedding tokens
        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"-{3,}", "---", line)
    return line


def to_url(rel_path: Path) -> str:
    parts = rel_path.with_suffix("").parts
    if parts[-1] == "index":
        parts = parts[:-1]
    return f"{DOCS_BASE_URL}/{'/'.join(parts)}/".replace("//", "/").replace(":/", "://")


def split_sections(md: str):
    """Yield (heading_path, lines) for each h1-h3 section; code fences are never split."""
    md = IFRAME.sub("", FRONT_MATTER.sub("", md, count=1))
    stack: dict[int, str] = {}
    lines: list[str] = []
    in_code = False

    def path():
        return " > ".join(stack[k] for k in sorted(stack))

    for raw in md.splitlines():
        if raw.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(raw)
            continue
        if in_code:
            lines.append(raw)
            continue
        m = HEADING.match(raw)
        if m:
            if lines:
                yield path(), lines
            level = len(m.group(1))
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = clean_prose(m.group(2)).strip()
            lines = []
        else:
            lines.append(clean_prose(raw))
    if lines:
        yield path(), lines


def to_blocks(lines: list[str]) -> list[str]:
    """Group lines into paragraphs; a fenced code block is always one block."""
    blocks, cur, in_code = [], [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        cur.append(line)
        if not in_code and not line.strip():
            blocks.append("\n".join(cur).strip())
            cur = []
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def split_big(block: str) -> list[str]:
    """Split an oversized block on line boundaries; table pieces repeat the header rows."""
    lines = block.splitlines()
    is_table = len(lines) > 2 and lines[0].lstrip().startswith("|") and bool(TABLE_SEP.match(lines[1]))
    limit = TABLE_CHARS if is_table else 2 * MAX_CHARS
    if len(block) <= limit:
        return [block]
    header = lines[:2] if is_table else []
    body = lines[2:] if is_table else lines
    size = TABLE_CHARS if is_table else MAX_CHARS
    out, cur = [], list(header)
    for line in body:
        if len(cur) > len(header) and len("\n".join(cur)) + len(line) + 1 > size:
            out.append("\n".join(cur))
            cur = list(header)
        cur.append(line)
    if len(cur) > len(header):
        out.append("\n".join(cur))
    return out


def pack(blocks: list[str]) -> list[str]:
    """Greedily pack blocks into pieces of at most MAX_CHARS (a lone big block stays whole)."""
    blocks = [p for b in blocks for p in split_big(b)]
    out, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > MAX_CHARS:
            out.append(cur)
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        out.append(cur)
    return out


def chunk_file(file: Path, docs_root: Path) -> list[Chunk]:
    rel = file.relative_to(docs_root)
    url = to_url(rel)
    md = expand_includes(file.read_text(encoding="utf-8"), docs_root.parent)
    chunks = []
    for heading, lines in split_sections(md):
        for piece in pack(to_blocks(lines)):
            if len(piece) < MIN_CHARS:
                continue
            text = f"{heading}\n\n{piece}" if heading else piece
            chunks.append(Chunk(f"{rel.as_posix()}#{len(chunks)}", text, rel.as_posix(), url, heading))
    return chunks
