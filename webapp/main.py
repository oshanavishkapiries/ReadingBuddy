import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from webapp.config import WORKSPACE, DEFAULT_EXTRACTION, DEFAULT_TRANSLATION, DEFAULT_PDF_GENERATION, OPENROUTER_API_KEY
from webapp.database import get_db, init_db
from webapp.models import create_job, get_job, list_jobs, update_job_status
from webapp.tasks import start_job, get_job_status

app = FastAPI(title="ReadingBuddy")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_extraction": DEFAULT_EXTRACTION,
        "default_translation": DEFAULT_TRANSLATION,
        "default_pdf_generation": DEFAULT_PDF_GENERATION,
        "has_api_key": bool(OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")),
    })


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    dpi: int = Form(300),
    ocr_mode: str = Form("auto"),
    lang: str = Form("eng"),
    model: str = Form("openai/gpt-4o-mini"),
    temperature: float = Form(0.2),
    api_key: str = Form(""),
    page_size: str = Form("A4"),
    margin: str = Form("18mm"),
    db: Session = Depends(get_db),
):
    upload_dir = WORKSPACE / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".pdf"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = upload_dir / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    settings = {
        "extraction": {
            "dpi": dpi,
            "ocr_mode": ocr_mode,
            "lang": lang,
            "save_page_render": True,
        },
        "translation": {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
        },
        "pdf_generation": {
            "page_size": page_size,
            "margin": margin,
        },
    }

    job = create_job(db, filename=file.filename, settings=settings)
    start_job(job.id, str(saved_path), settings)

    return RedirectResponse(url=f"/job/{job.id}", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request, db: Session = Depends(get_db)):
    jobs = list_jobs(db)
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        return HTMLResponse("Job not found", status_code=404)
    return templates.TemplateResponse("job_detail.html", {"request": request, "job": job})


@app.get("/job/{job_id}/status")
async def job_status(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        return {"error": "not found"}
    live = get_job_status(job_id)
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "progress": live["progress"] if live else job.progress,
        "current_step": live["current_step"] if live else job.current_step,
        "step_detail": live["step_detail"] if live else job.step_detail,
        "error": job.error,
        "output_pdf": job.output_pdf,
        "page_count": job.page_count,
    }


@app.get("/job/{job_id}/download")
async def download_pdf(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job or not job.output_pdf:
        return HTMLResponse("PDF not available", status_code=404)
    pdf_path = Path(job.output_pdf)
    if not pdf_path.exists():
        return HTMLResponse("PDF file not found", status_code=404)
    return FileResponse(
        path=str(pdf_path),
        filename=f"readingbuddy_{job_id}.pdf",
        media_type="application/pdf",
    )


@app.get("/job/{job_id}/download/markdown")
async def download_markdown(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        return HTMLResponse("Job not found", status_code=404)
    markdown_dir = WORKSPACE / job_id / "markdown"
    if not markdown_dir.exists():
        return HTMLResponse("Markdown not available", status_code=404)
    combined = markdown_dir / "combined.md"
    if combined.exists():
        return FileResponse(
            path=str(combined),
            filename=f"readingbuddy_{job_id}_markdown.md",
            media_type="text/markdown",
        )
    return HTMLResponse("Combined markdown not found", status_code=404)


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        return {"error": "not found"}
    if job.status in ("pending", "processing"):
        update_job_status(db, job_id, "cancelled", error="Cancelled by user")
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/health")
async def health():
    return {"status": "ok"}
