from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import markdown

if TYPE_CHECKING:
    from playwright.sync_api import sync_playwright


def natural_sort_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def make_css(font_family: str, font_file: Optional[Path], page_size: str, margin: str, font_size: float = 16.5) -> str:
    font_face = ""
    if font_file is not None and font_file.exists():
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
  font-size: {font_size}pt;
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
        extensions=["extra", "tables", "sane_lists", "nl2br"],
        output_format="html5",
    )


def build_html(markdown_dir: Path, css: str, image_root: Optional[Path]) -> str:
    md_files = sorted(markdown_dir.glob("page_*.md"), key=natural_sort_key)
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


def render_pdf(html_path: Path, pdf_path: Path, page_size: str, progress_callback: Callable[[float, str, str], None] | None = None):
    from playwright.sync_api import sync_playwright
    if progress_callback:
        progress_callback(30, "Generating PDF", "Launching browser...")
    with sync_playwright() as p:
        if progress_callback:
            progress_callback(50, "Generating PDF", "Loading content...")
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        if progress_callback:
            progress_callback(80, "Generating PDF", "Rendering PDF...")
        page.pdf(
            path=str(pdf_path.resolve()),
            format=page_size,
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
        )
        browser.close()
        if progress_callback:
            progress_callback(100, "Generating PDF", "Complete!")


def generate_pdf(
    markdown_dir: str,
    output_pdf: str,
    image_root: Optional[str] = None,
    font_file: Optional[str] = None,
    font_family: str = "SinhalaFont",
    font_size: float = 16.5,
    page_size: str = "A4",
    margin: str = "18mm",
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> str:
    markdown_path = Path(markdown_dir).resolve()
    output_path = Path(output_pdf).resolve()
    image_path = Path(image_root).resolve() if image_root else None
    font_path = Path(font_file) if font_file else None

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown folder not found: {markdown_path}")

    if font_path and not font_path.exists():
        font_path = None

    if progress_callback:
        progress_callback(10, "Generating PDF", "Building HTML...")

    css = make_css(
        font_family=font_family,
        font_file=font_path,
        page_size=page_size,
        margin=margin,
        font_size=font_size,
    )

    full_html = build_html(
        markdown_dir=markdown_path,
        css=css,
        image_root=image_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "print.html"
        html_path.write_text(full_html, encoding="utf-8")
        render_pdf(html_path, output_path, page_size, progress_callback)

    return str(output_path)
