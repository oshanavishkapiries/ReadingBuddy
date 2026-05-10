import threading
import traceback
from pathlib import Path
from typing import Dict

from sqlalchemy.orm import Session

from webapp.config import WORKSPACE, DEFAULT_EXTRACTION, DEFAULT_TRANSLATION, DEFAULT_PDF_GENERATION
from webapp.database import SessionLocal
from webapp.models import update_job_progress, update_job_status
from webapp.pipeline import extractor, translator, pdf_generator

active_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def make_progress_callback(job_id: str):
    def callback(progress: float, step: str, detail: str):
        db = SessionLocal()
        try:
            update_job_progress(db, job_id, progress, step, detail)
            with _lock:
                if job_id in active_jobs:
                    active_jobs[job_id]["progress"] = progress
                    active_jobs[job_id]["current_step"] = step
                    active_jobs[job_id]["step_detail"] = detail
        finally:
            db.close()
    return callback


def run_pipeline(job_id: str, pdf_path: str, settings: dict):
    db = SessionLocal()
    try:
        update_job_progress(db, job_id, 0, "Starting", "Initializing pipeline...")
        with _lock:
            active_jobs[job_id] = {"progress": 0, "current_step": "Starting", "step_detail": "Initializing..."}

        job_workspace = WORKSPACE / job_id
        extract_dir = job_workspace / "extracted"
        markdown_dir = job_workspace / "markdown"
        output_pdf = job_workspace / "final.pdf"

        ext = settings.get("extraction", DEFAULT_EXTRACTION)
        trans = settings.get("translation", DEFAULT_TRANSLATION)
        pdf_gen = settings.get("pdf_generation", DEFAULT_PDF_GENERATION)

        api_key = trans.get("api_key", "")
        if not api_key:
            raise ValueError("OpenRouter API key is not configured. Please add it in Settings.")

        callback = make_progress_callback(job_id)

        update_job_progress(db, job_id, 5, "Extracting PDF", "Starting extraction...")
        manifest = extractor.extract_pdf(
            pdf_path=pdf_path,
            output_dir=str(extract_dir),
            dpi=ext.get("dpi", 300),
            ocr_mode=ext.get("ocr_mode", "auto"),
            lang=ext.get("lang", "eng"),
            min_text_chars=ext.get("min_text_chars", 80),
            psm=ext.get("psm", 6),
            min_image_area=ext.get("min_image_area", 2500),
            save_page_render=ext.get("save_page_render", True),
            progress_callback=callback,
        )

        page_count = manifest.get("pages", 0)
        update_job_progress(db, job_id, 5, "Extraction complete", f"Extracted {page_count} pages")
        update_job_progress(db, job_id, 10, "Translating", "Starting translation...")

        translator.translate_to_sinhala(
            input_dir=str(extract_dir),
            output_dir=str(markdown_dir),
            api_key=api_key,
            model=trans.get("model", DEFAULT_TRANSLATION["model"]),
            temperature=trans.get("temperature", 0.2),
            max_tokens=trans.get("max_tokens", 0),
            retries=trans.get("retries", 3),
            timeout=trans.get("timeout", 120),
            continue_on_error=trans.get("continue_on_error", True),
            progress_callback=callback,
        )

        update_job_progress(db, job_id, 5, "Translation complete", "Generating PDF...")

        font_file = pdf_gen.get("font_file", DEFAULT_PDF_GENERATION["font_file"])
        pdf_generator.generate_pdf(
            markdown_dir=str(markdown_dir),
            output_pdf=str(output_pdf),
            image_root=str(extract_dir),
            font_file=font_file if Path(font_file).exists() else None,
            font_family=pdf_gen.get("font_family", "SinhalaFont"),
            page_size=pdf_gen.get("page_size", "A4"),
            margin=pdf_gen.get("margin", "18mm"),
            progress_callback=callback,
        )

        update_job_status(db, job_id, "completed", output_pdf=str(output_pdf))
        with _lock:
            active_jobs.pop(job_id, None)

    except Exception as e:
        update_job_status(db, job_id, "failed", error=str(e))
        with _lock:
            active_jobs.pop(job_id, None)
        traceback.print_exc()
    finally:
        db.close()


def start_job(job_id: str, pdf_path: str, settings: dict):
    thread = threading.Thread(target=run_pipeline, args=(job_id, pdf_path, settings), daemon=True)
    thread.start()
    return thread


def get_job_status(job_id: str) -> dict | None:
    with _lock:
        return active_jobs.get(job_id)
