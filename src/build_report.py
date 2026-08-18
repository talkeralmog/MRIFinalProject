# Authors: Michal Yechezkel (ID: 322556267), Almog Talker (ID: 322546680)
"""Build the final project report as a PDF and as a DOCX.

The document body lives in ``src/report_content.py`` as a list of blocks and every number
in it comes from ``src/report_data.py``, which reads the logged results. Regenerating the
report after re-running an experiment therefore needs no editing.

Usage::

    python -m src.build_report
    python -m src.build_report --github https://github.com/user/repo --formats pdf
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

from .report_content import GITHUB_URL, build_blocks
from .report_data import Results
from .report_render import plain_text, render_docx, render_pdf


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-root", default="results")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--out-dir", default="report")
    p.add_argument("--stem", default="MRI_Final_Project_Report")
    p.add_argument("--github", default=GITHUB_URL,
                   help="Public repository URL printed on the title page.")
    p.add_argument("--formats", nargs="*", default=["pdf", "docx", "txt"],
                   choices=["pdf", "docx", "txt"])
    return p.parse_args(argv)


def main(argv=None) -> List[str]:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    R = Results(args.results_root, args.cache_root)
    blocks = build_blocks(R, args.github)

    written: List[str] = []
    if "pdf" in args.formats:
        path = os.path.join(args.out_dir, f"{args.stem}.pdf")
        render_pdf(blocks, path)
        written.append(path)
    if "docx" in args.formats:
        path = os.path.join(args.out_dir, f"{args.stem}.docx")
        render_docx(blocks, path)
        written.append(path)
    if "txt" in args.formats:
        # A plain-text dump of the prose, used by the review passes and for diffing.
        path = os.path.join(args.out_dir, f"{args.stem}.txt")
        with open(path, "w") as f:
            for b in blocks:
                kind = b["type"]
                if kind == "title":
                    f.write(f"{plain_text(b['course'])}\nFinal Project\n\n"
                            f"{plain_text(b['heading'])}\n{plain_text(b['subheading'])}\n"
                            f"{plain_text(b['authors'])}\n{plain_text(b['link'])}\n\n")
                elif kind in ("h1", "h2", "h3"):
                    prefix = {"h1": "\n\n## ", "h2": "\n### ", "h3": "\n#### "}[kind]
                    f.write(f"{prefix}{plain_text(b['text'])}\n\n")
                elif kind == "p":
                    f.write(plain_text(b["text"]) + "\n\n")
                elif kind == "bullets":
                    for item in b["items"]:
                        f.write("- " + plain_text(item) + "\n")
                    f.write("\n")
                elif kind == "callout":
                    f.write(f"[BOX] {plain_text(b['title'])}\n"
                            f"{plain_text(b['body'])}\n\n")
                elif kind == "table":
                    for row in b["rows"]:
                        f.write(" | ".join(plain_text(str(c)) for c in row) + "\n")
                    if b.get("caption"):
                        f.write(plain_text(b["caption"]) + "\n")
                    f.write("\n")
                elif kind == "figure":
                    f.write(f"[FIGURE {os.path.basename(b['path'])}] "
                            f"{plain_text(b.get('caption', ''))}\n\n")
        written.append(path)

    for path in written:
        size = os.path.getsize(path) / 1024.0
        print(f"wrote {path}  ({size:.0f} KB)")

    check_pdf_glyphs(os.path.join(args.out_dir, f"{args.stem}.pdf"))
    return written


# ---------------------------------------------------------------------------
# Post-build check
# ---------------------------------------------------------------------------

#: Characters that mean a glyph was missing from the font rather than intended.
NOTDEF = ("\u25a0", "\u25a1", "\ufffd")   # black box, white box, replacement char


def check_pdf_glyphs(pdf_path: str) -> List[str]:
    """Warn if the built PDF contains a missing-glyph box.

    The base-14 Times font has no glyph for some characters we might reasonably write
    (U+2113 SCRIPT SMALL L, for one), and ReportLab substitutes a black box rather than
    failing. That is invisible unless the page is inspected, so the text is extracted
    once after every build and checked. Silently shipping a box is the failure this
    prevents.
    """
    if not os.path.exists(pdf_path):
        return []
    try:
        text = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True,
                              text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []   # pdftotext unavailable; skip rather than fail the build

    problems: List[str] = []
    for line_no, line in enumerate(text.split("\n"), 1):
        for char in NOTDEF:
            if char in line:
                problems.append(f"line {line_no}: {line.strip()[:90]}")
                break
    if problems:
        print(f"\nWARNING: {len(problems)} missing-glyph box(es) in {pdf_path}:")
        for problem in problems[:10]:
            print(f"  {problem}")
        print("  Rewrite the offending character; see the note on ENTITIES in "
              "src/report_render.py.")
    else:
        print("glyph check: no missing-glyph boxes in the PDF")
    return problems


if __name__ == "__main__":
    main()
