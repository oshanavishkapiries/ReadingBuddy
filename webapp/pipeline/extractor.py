from __future__ import annotations

import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    import fitz

from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


def _get_fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        raise RuntimeError("pymupdf is not installed. Run: pip install pymupdf")


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text[:max_len].strip("_") or "file"


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    fitz = _get_fitz()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def extract_digital_text(page: fitz.Page) -> Tuple[str, List[Dict[str, Any]]]:
    text = page.get_text("text", sort=True) or ""
    words_raw = page.get_text("words", sort=True) or []
    words = []
    for w in words_raw:
        words.append({
            "text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
            "block": w[5], "line": w[6], "word": w[7]
        })
    return text.strip(), words


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    if cv2 is not None and np is not None:
        arr = np.array(gray)
        arr = cv2.fastNlMeansDenoising(arr, None, 12, 7, 21)
        arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )
        return Image.fromarray(arr)
    return gray


def run_ocr(img: Image.Image, lang: str, psm: int) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed")
    processed = preprocess_for_ocr(img)
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(processed, lang=lang, config=config).strip()


def crop_rect_from_render(page: fitz.Page, page_img: Image.Image, rect: fitz.Rect, pad: int = 4) -> Image.Image:
    fitz = _get_fitz()
    scale_x = page_img.width / float(page.rect.width)
    scale_y = page_img.height / float(page.rect.height)
    x0 = max(0, int(rect.x0 * scale_x) - pad)
    y0 = max(0, int(rect.y0 * scale_y) - pad)
    x1 = min(page_img.width, int(rect.x1 * scale_x) + pad)
    y1 = min(page_img.height, int(rect.y1 * scale_y) + pad)
    return page_img.crop((x0, y0, x1, y1))


def merge_nearby_rects(rects: List[fitz.Rect], max_gap: float = 25) -> List[fitz.Rect]:
    fitz = _get_fitz()
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


def looks_like_image(crop: Image.Image, min_variance: float = 500.0) -> bool:
    if np is None:
        return True
    gray = np.array(crop.convert("L"), dtype=float)
    return float(np.var(gray)) > min_variance


def save_image_blocks(page: fitz.Page, page_img: Image.Image, out_dir: Path, min_area: int) -> int:
    fitz = _get_fitz()
    count = 0
    data = page.get_text("dict")
    page_area = page.rect.width * page.rect.height
    candidate_rects: List[fitz.Rect] = []

    for block in data.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox:
            continue
        if block.get("type") == 1:  # image blocks only; type=0 are text blocks
            candidate_rects.append(fitz.Rect(bbox))

    try:
        drawings = page.get_drawings()
        for drawing in drawings:
            rect = drawing.get("rect")
            if not rect:
                continue
            rect = fitz.Rect(rect)
            area = rect.width * rect.height
            if area < page_area * 0.01:  # skip tiny decorative elements
                continue
            aspect = rect.width / max(rect.height, 1)
            if aspect > 15 or aspect < 0.07:  # skip thin rules and separators
                continue
            candidate_rects.append(rect)
    except Exception:
        pass

    if not candidate_rects:
        return 0

    merged_rects = merge_nearby_rects(candidate_rects, max_gap=10)
    for rect in merged_rects:
        rect = fitz.Rect(rect)
        rect.x0 = max(page.rect.x0, rect.x0 - 12)
        rect.y0 = max(page.rect.y0, rect.y0 - 12)
        rect.x1 = min(page.rect.x1, rect.x1 + 12)
        rect.y1 = min(page.rect.y1, rect.y1 + 12)
        crop = crop_rect_from_render(page, page_img, rect, pad=6)
        if crop.width * crop.height < min_area:
            continue
        if crop.width > page_img.width * 0.95 and crop.height > page_img.height * 0.95:
            continue
        if crop.height < 50:  # skip very thin strips
            continue
        if not looks_like_image(crop):  # skip text-looking regions
            continue
        count += 1
        crop.save(out_dir / f"crop_{count:03d}.png")
    return count


def save_embedded_images(doc: fitz.Document, page: fitz.Page, out_dir: Path, page_number: int) -> int:
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


def extract_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    ocr_mode: str = "auto",
    lang: str = "eng",
    min_text_chars: int = 80,
    psm: int = 6,
    min_image_area: int = 2500,
    save_page_render: bool = True,
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> dict:
    fitz = _get_fitz()
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    manifest: Dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "pages": total_pages,
        "dpi": dpi,
        "ocr_mode": ocr_mode,
        "language": lang,
        "page_outputs": []
    }

    for page_index in range(total_pages):
        page_no = page_index + 1
        page = doc[page_index]
        page_dir = out_root / f"page_{page_no:03d}"
        image_dir = page_dir / "images"
        page_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        digital_text, words = extract_digital_text(page)
        (page_dir / "digital_text.txt").write_text(digital_text + "\n", encoding="utf-8")
        (page_dir / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")

        page_img = render_page(page, dpi)
        if save_page_render:
            page_img.save(page_dir / "page.png")

        should_ocr = ocr_mode == "always" or (ocr_mode == "auto" and len(digital_text) < min_text_chars)
        ocr_text = ""
        if should_ocr:
            try:
                ocr_text = run_ocr(page_img, lang, psm)
            except Exception as exc:
                print(f"Warning: OCR failed on page {page_no}: {exc}", file=sys.stderr)
        (page_dir / "ocr_text.txt").write_text(ocr_text + "\n", encoding="utf-8")

        best_text = choose_best_text(digital_text, ocr_text, min_text_chars)
        (page_dir / "text.txt").write_text(best_text + "\n", encoding="utf-8")

        embedded_count = save_embedded_images(doc, page, image_dir, page_no)
        # Only crop for vector diagrams when the PDF has no embedded images on this page
        crop_count = save_image_blocks(page, page_img, image_dir, min_image_area) if embedded_count == 0 else 0

        manifest["page_outputs"].append({
            "page": page_no,
            "folder": str(page_dir),
            "digital_chars": len(digital_text),
            "ocr_chars": len(ocr_text),
            "image_crops": crop_count,
            "embedded_images": embedded_count,
        })

        progress = ((page_index + 1) / total_pages) * 100
        detail = f"Page {page_no}/{total_pages}: text={len(best_text)} chars, crops={crop_count}, embedded={embedded_count}"
        if progress_callback:
            progress_callback(progress, "Extracting PDF pages", detail)

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close()
    return manifest
