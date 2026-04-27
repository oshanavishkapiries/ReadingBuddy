#!/usr/bin/env python3

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import markdown


def natural_sort_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def make_css(font_family: str, font_file: Optional[Path], page_size: str, margin: str) -> str:
    font_face = ""
    if font_file is not None:
        font_face = f"""
@font-face {{
  font-family: '{font_family}';
  src: url('{font_file.resolve().as_uri()}') format('truetype');
  font-weight: 400;
  font-style: normal;
}}
"""

    return f"""
{font_face}

@page {{
  size: {page_size};
  margin: {margin};
}}

html, body {{
  font-family: '{font_family}', 'Noto Serif Sinhala', 'Noto Sans Sinhala', sans-serif;
  font-size: 16.5pt;
  line-height: 1.6;
  color: #000;
  background: #fff;
  -webkit-font-smoothing: subpixel-antialiased;
  text-rendering: optimizeLegibility;
}}

body, h1, h2, h3, h4, h5, h6, p, div, span, li, td, th, a, strong, em {{
  font-family: '{font_family}', 'Noto Serif Sinhala', 'Noto Sans Sinhala', sans-serif !important;
}}

h1, h2, h3, h4 {{
  line-height: 1.35;
  margin: 0.75em 0 0.4em;
  font-weight: 700;
  color: #000;
}}

h1 {{ font-size: 20pt; }}
h2 {{ font-size: 16pt; }}
h3 {{ font-size: 13.5pt; }}

p {{
  margin: 0.4em 0;
}}

ul, ol {{
  margin-top: 0.3em;
}}

li {{
  margin: 0.2em 0;
}}

strong, b {{
  font-weight: 700;
}}

img {{
  max-width: 100%;
  max-height: 720px;
  height: auto;
  display: block;
  margin: 10px auto;
  break-inside: avoid;
}}

table {{
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0;
  font-size: 10pt;
}}

th, td {{
  border: 1px solid #666;
  padding: 5px 7px;
  vertical-align: top;
}}

blockquote {{
  border-left: 4px solid #999;
  margin-left: 0;
  padding-left: 10px;
  color: #222;
}}

code, pre {{
  font-family: monospace;
}}

pre {{
  white-space: pre-wrap;
  border: 1px solid #ddd;
  padding: 8px;
}}

.page-break {{
  page-break-before: always;
}}
"""


def fix_image_paths(markdown_text: str, image_root: Optional[Path]) -> str:
    if image_root is None:
        return markdown_text

    image_root = image_root.resolve()

    def replace(match):
        alt = match.group(1)
        src = match.group(2).strip()

        if src.startswith(("http://", "https://", "file://", "data:")):
            return match.group(0)

        src_path = Path(src)

        candidates = [
            Path(src).resolve(),
            image_root / src_path,
            image_root.parent / src_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return f"![{alt}]({candidate.resolve().as_uri()})"

        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown_text)


def markdown_to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=[
            "extra",
            "tables",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )


def build_html(markdown_dir: Path, css: str, image_root: Optional[Path]) -> str:
    md_files = sorted(markdown_dir.glob("*.md"), key=natural_sort_key)

    if not md_files:
        raise FileNotFoundError(f"No .md files found in {markdown_dir}")

    parts = []

    for index, md_file in enumerate(md_files):
        md_text = md_file.read_text(encoding="utf-8")
        md_text = fix_image_paths(md_text, image_root)
        page_html = markdown_to_html(md_text)

        if index > 0:
            parts.append('<div class="page-break"></div>')

        parts.append(page_html)

    body = "\n".join(parts)

    return f"""<!doctype html>
<html lang="si">
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def find_chromium() -> str:
    candidates = [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
    ]

    for name in candidates:
        path = shutil.which(name)
        if path:
            return path

    raise RuntimeError(
        "Chromium/Chrome not found. Install it with:\n"
        "sudo apt-get update && sudo apt-get install -y chromium\n"
        "or\n"
        "sudo apt-get install -y chromium-browser"
    )


def html_to_pdf_with_chromium(html_path: Path, pdf_path: Path, page_size: str):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")

        page.pdf(
            path=str(pdf_path.resolve()),
            format=page_size,
            print_background=True,
            prefer_css_page_size=True,
            margin={
                "top": "0mm",
                "right": "0mm",
                "bottom": "0mm",
                "left": "0mm",
            },
        )

        browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sinhala Markdown pages to sharp PDF using Chromium."
    )
    parser.add_argument(
        "markdown_dir",
        type=Path,
        help="Folder containing page_001.md, page_002.md, etc.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("final.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root folder containing extracted page image folders.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Sinhala font file, e.g. NotoSerifSinhala-Regular.ttf",
    )
    parser.add_argument(
        "--font-family",
        default="SinhalaFont",
        help="Internal CSS font-family name.",
    )
    parser.add_argument(
        "--page-size",
        default="A4",
        help="PDF page size, e.g. A4, Letter.",
    )
    parser.add_argument(
        "--margin",
        default="18mm",
        help="PDF margin, e.g. 18mm, 20mm.",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="Save generated HTML next to output PDF.",
    )

    args = parser.parse_args()

    markdown_dir = args.markdown_dir.resolve()
    output_pdf = args.output.resolve()

    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown folder not found: {markdown_dir}")

    if args.font_file is not None and not args.font_file.exists():
        raise FileNotFoundError(f"Font file not found: {args.font_file}")

    css = make_css(
        font_family=args.font_family,
        font_file=args.font_file,
        page_size=args.page_size,
        margin=args.margin,
    )

    full_html = build_html(
        markdown_dir=markdown_dir,
        css=css,
        image_root=args.image_root,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_html:
        html_path = output_pdf.with_suffix(".html")
        html_path.write_text(full_html, encoding="utf-8")
        html_to_pdf_with_chromium(html_path, output_pdf, args.page_size)
        print(f"HTML saved: {html_path}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "print.html"
            html_path.write_text(full_html, encoding="utf-8")
            html_to_pdf_with_chromium(html_path, output_pdf, args.page_size)

    print(f"PDF saved: {output_pdf}")


if __name__ == "__main__":
    main()