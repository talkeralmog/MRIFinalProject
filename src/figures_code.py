# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Render short excerpts of our own source as syntax-highlighted listings.

The report shows the three pieces of code that carry the MRI content of the project --
the undersampling mask, the classical baseline's iteration, and the learned
data-consistency step -- so a reader can check the claims against the implementation
without leaving the document. Excerpts are pulled from the live source files by function
name, so they cannot go stale if the code changes.

Usage::

    python -m src.figures_code
"""

from __future__ import annotations

import argparse
import ast
import os
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

# (output stem, source file, object path, strip the docstring?)
LISTINGS: Sequence[Tuple[str, str, str, bool]] = (
    ("code_mask", "src/masks.py", "gaussian1d_mask", True),
    ("code_baseline", "src/baselines/classical_cs.py", "ClassicalCS.forward", False),
    ("code_stage", "src/model.py", "CustomADMMStage.data_consistency", False),
    ("code_forward", "src/model.py", "CustomADMMStage.forward", False),
)


def extract(source_path: str, dotted: str, drop_docstring: bool = False) -> str:
    """Return the source of ``dotted`` (``Class.method`` or ``function``) verbatim.

    With ``drop_docstring`` the leading docstring is removed, which keeps a long,
    heavily-documented function short enough to show as one listing.
    """
    text = open(source_path).read()
    tree = ast.parse(text)
    parts = dotted.split(".")

    node: Optional[ast.AST] = None
    scope: Sequence[ast.AST] = tree.body
    for part in parts:
        node = None
        for candidate in scope:
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and candidate.name == part:
                node = candidate
                break
        if node is None:
            raise KeyError(f"{dotted!r} not found in {source_path}")
        scope = getattr(node, "body", [])

    lines = text.splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = node.end_lineno

    if drop_docstring and ast.get_docstring(node):
        doc_node = node.body[0]
        keep = (lines[start - 1 : doc_node.lineno - 1]
                + lines[doc_node.end_lineno : end])
        snippet = "\n".join(keep)
    else:
        snippet = "\n".join(lines[start - 1 : end])
    return textwrap.dedent(snippet).rstrip() + "\n"


def render(code: str, out_path: str, font_size: int = 15) -> str:
    """Write ``code`` as a syntax-highlighted PNG listing."""
    formatter = ImageFormatter(
        font_size=font_size, line_numbers=False, style="friendly",
        line_pad=3, image_pad=12,
        font_name="Menlo" if os.path.exists("/System/Library/Fonts/Menlo.ttc")
        else "DejaVu Sans Mono",
    )
    png = highlight(code, PythonLexer(), formatter)
    with open(out_path, "wb") as f:
        f.write(png)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=os.path.join("results", "figures"))
    p.add_argument("--font-size", type=int, default=15)
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    written: List[str] = []
    for stem, source, dotted, strip_doc in LISTINGS:
        code = extract(source, dotted, strip_doc)
        path = os.path.join(args.out_dir, f"{stem}.png")
        render(code, path, args.font_size)
        written.append(path)
        print(f"wrote {path}  ({dotted} from {source}, {len(code.splitlines())} lines)")
    return written


if __name__ == "__main__":
    main()
