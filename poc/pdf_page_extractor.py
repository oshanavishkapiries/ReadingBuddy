#!/usr/bin/env python3
"""
PDF page extractor: per-page folders with strong text extraction + image crops.

Output structure:
  output_dir/
    page_001/
      text.txt              # best text: digital text + OCR fallback/optional OCR merge
      digital_text.txt      # text extracted directly from PDF layer
      ocr_text.txt          # OCR text if OCR ran
      words.json            # word-level boxes from PDF text layer when available
      page.png              # rendered page image, optional
      images/
        crop_001.png        # visual crops using PDF image block bboxes
        embedded_xref_...   # raw embedded image stream export

Install:
  pip install pymupdf pillow pytesseract opencv-python
  # Also install Tesseract OCR binary:
  # Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-eng
  # macOS: brew install tesseract

Usage:
  python pdf_page_extractor.py input.pdf -o extracted --lang eng --dpi 300
  python pdf_page_extractor.py input.pdf -o extracted --ocr-mode always --save-page-render
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text[:max_len].strip("_") or "file"


def ensure_clean_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def extract_digital_text(page: fitz.Page) -> Tuple[str, List[Dict[str, Any]]]:
    # sort=True improves reading order for many normal PDFs.
    text = page.get_text("text", sort=True) or ""
    words_raw = page.get_text("words", sort=True) or []
    words = []
    for w in words_raw:
        # x0, y0, x1, y1, word, block_no, line_no, word_no
        words.append({
            "text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
            "block": w[5], "line": w[6], "word": w[7]
        })
    return text.strip(), words


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Conservative preprocessing that improves English OCR without damaging layout too much."""
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)

    if cv2 is not None and np is not None:
        arr = np.array(gray)
        # Denoise then adaptive threshold for scans/photos.
        arr = cv2.fastNlMeansDenoising(arr, None, 12, 7, 21)
        arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )
        return Image.fromarray(arr)
    return gray


def run_ocr(img: Image.Image, lang: str, psm: int) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")
    processed = preprocess_for_ocr(img)
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(processed, lang=lang, config=config).strip()


def crop_rect_from_render(page: fitz.Page, page_img: Image.Image, rect: fitz.Rect, pad: int = 4) -> Image.Image:
    scale_x = page_img.width / float(page.rect.width)
    scale_y = page_img.height / float(page.rect.height)
    x0 = max(0, int(rect.x0 * scale_x) - pad)
    y0 = max(0, int(rect.y0 * scale_y) - pad)
    x1 = min(page_img.width, int(rect.x1 * scale_x) + pad)
    y1 = min(page_img.height, int(rect.y1 * scale_y) + pad)
    return page_img.crop((x0, y0, x1, y1))


def merge_nearby_rects(rects: List[fitz.Rect], max_gap: float = 25) -> List[fitz.Rect]:
    """Merge nearby/overlapping rectangles so diagrams are saved as one image."""
    merged: List[fitz.Rect] = []

    for rect in rects:
        rect = fitz.Rect(rect)

        did_merge = False
        for i, existing in enumerate(merged):
            expanded = fitz.Rect(existing)
            expanded.x0 -= max_gap
            expanded.y0 -= max_gap
            expanded.x1 += max_gap
            expanded.y1 += max_gap

            if expanded.intersects(rect):
                merged[i] = existing | rect
                did_merge = True
                break

        if not did_merge:
            merged.append(rect)

    # second pass merge
    changed = True
    while changed:
        changed = False
        result: List[fitz.Rect] = []

        for rect in merged:
            merged_into_existing = False
            for i, existing in enumerate(result):
                expanded = fitz.Rect(existing)
                expanded.x0 -= max_gap
                expanded.y0 -= max_gap
                expanded.x1 += max_gap
                expanded.y1 += max_gap

                if expanded.intersects(rect):
                    result[i] = existing | rect
                    changed = True
                    merged_into_existing = True
                    break

            if not merged_into_existing:
                result.append(rect)

        merged = result

    return merged


def save_image_blocks(page: fitz.Page, page_img: Image.Image, out_dir: Path, min_area: int) -> int:
    """
    Save normal PDF image blocks + large diagram-like visual regions.

    This fixes pages where diagrams are made from text, lines, arrows, and shapes,
    not from one embedded image.
    """
    count = 0
    data = page.get_text("dict")
    page_area = page.rect.width * page.rect.height

    candidate_rects: List[fitz.Rect] = []

    # 1) Normal embedded image blocks
    for block in data.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox:
            continue

        rect = fitz.Rect(bbox)
        area = rect.width * rect.height

        if block.get("type") == 1:
            candidate_rects.append(rect)

        # 2) Large text/layout blocks can be diagram labels/captions inside figures
        # This catches diagrams where PDF stores labels as text objects.
        elif block.get("type") == 0:
            if area > page_area * 0.025 and rect.width > page.rect.width * 0.20:
                candidate_rects.append(rect)

    # 3) Add vector drawings / shapes / arrows / boxes
    try:
        drawings = page.get_drawings()
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect:
                rect = fitz.Rect(rect)
                if rect.width * rect.height > page_area * 0.001:
                    candidate_rects.append(rect)
    except Exception:
        pass

    if not candidate_rects:
        return 0

    # Merge diagram parts into larger full-diagram crops
    merged_rects = merge_nearby_rects(candidate_rects, max_gap=35)

    for rect in merged_rects:
        # Add padding around diagram
        rect = fitz.Rect(rect)
        rect.x0 = max(page.rect.x0, rect.x0 - 12)
        rect.y0 = max(page.rect.y0, rect.y0 - 12)
        rect.x1 = min(page.rect.x1, rect.x1 + 12)
        rect.y1 = min(page.rect.y1, rect.y1 + 12)

        crop = crop_rect_from_render(page, page_img, rect, pad=6)

        if crop.width * crop.height < min_area:
            continue

        # Skip almost-full-page crops unless it is really a visual-heavy page
        if crop.width > page_img.width * 0.95 and crop.height > page_img.height * 0.95:
            continue

        count += 1
        crop.save(out_dir / f"crop_{count:03d}.png")

    return count

def save_embedded_images(doc: fitz.Document, page: fitz.Page, out_dir: Path, page_number: int) -> int:
    """Export raw embedded image streams. Useful when crops miss masks/transparent images."""
    count = 0
    seen = set()
    for info in page.get_images(full=True):
        xref = info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            image = doc.extract_image(xref)
            ext = image.get("ext", "png")
            data = image["image"]
            count += 1
            name = f"embedded_p{page_number:03d}_xref{xref}.{safe_name(ext)}"
            (out_dir / name).write_bytes(data)
        except Exception as exc:
            print(f"Warning: could not extract image xref {xref}: {exc}", file=sys.stderr)
    return count


def choose_best_text(digital: str, ocr: str, min_chars: int) -> str:
    if len(digital.strip()) >= min_chars and len(digital.strip()) >= max(30, len(ocr.strip()) * 0.45):
        if ocr.strip() and ocr.strip() not in digital:
            return digital.strip() + "\n\n--- OCR text ---\n" + ocr.strip()
        return digital.strip()
    return ocr.strip() or digital.strip()


def process_pdf(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    out_root = Path(args.output)
    ensure_clean_dir(out_root, args.clean)

    doc = fitz.open(pdf_path)
    manifest: Dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "pages": len(doc),
        "dpi": args.dpi,
        "ocr_mode": args.ocr_mode,
        "language": args.lang,
        "page_outputs": []
    }

    for page_index in range(len(doc)):
        page_no = page_index + 1
        page = doc[page_index]
        page_dir = out_root / f"page_{page_no:03d}"
        image_dir = page_dir / "images"
        page_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        digital_text, words = extract_digital_text(page)
        (page_dir / "digital_text.txt").write_text(digital_text + "\n", encoding="utf-8")
        (page_dir / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")

        page_img = render_page(page, args.dpi)
        if args.save_page_render:
            page_img.save(page_dir / "page.png")

        should_ocr = args.ocr_mode == "always" or (args.ocr_mode == "auto" and len(digital_text) < args.min_text_chars)
        ocr_text = ""
        if should_ocr:
            try:
                ocr_text = run_ocr(page_img, args.lang, args.psm)
            except Exception as exc:
                print(f"Warning: OCR failed on page {page_no}: {exc}", file=sys.stderr)
        (page_dir / "ocr_text.txt").write_text(ocr_text + "\n", encoding="utf-8")

        best_text = choose_best_text(digital_text, ocr_text, args.min_text_chars)
        (page_dir / "text.txt").write_text(best_text + "\n", encoding="utf-8")

        crop_count = save_image_blocks(page, page_img, image_dir, args.min_image_area)
        embedded_count = save_embedded_images(doc, page, image_dir, page_no)

        manifest["page_outputs"].append({
            "page": page_no,
            "folder": str(page_dir),
            "digital_chars": len(digital_text),
            "ocr_chars": len(ocr_text),
            "image_crops": crop_count,
            "embedded_images": embedded_count,
        })
        print(f"page {page_no:03d}: text={len(best_text)} chars, crops={crop_count}, embedded={embedded_count}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract each PDF page into its own folder with text and image crops.")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--output", default="pdf_extracted", help="Output folder")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR and crops. 300 is a strong default.")
    parser.add_argument("--lang", default="eng", help="Tesseract language, e.g. eng, eng+sin")
    parser.add_argument("--ocr-mode", choices=["auto", "always", "never"], default="auto",
                        help="auto = OCR only when digital text is weak; always = OCR every page")
    parser.add_argument("--min-text-chars", type=int, default=80,
                        help="If digital text has fewer chars than this, auto OCR runs")
    parser.add_argument("--psm", type=int, default=6,
                        help="Tesseract page segmentation mode. Try 3 for full pages, 6 for uniform text blocks.")
    parser.add_argument("--min-image-area", type=int, default=2500,
                        help="Skip tiny image crops below this pixel area")
    parser.add_argument("--save-page-render", action="store_true", help="Save rendered page.png in each page folder")
    parser.add_argument("--clean", action="store_true", help="Delete output folder before writing")
    return parser


if __name__ == "__main__":
    process_pdf(build_parser().parse_args())
