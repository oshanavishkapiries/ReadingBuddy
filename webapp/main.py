import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from webapp.config import WORKSPACE, DEFAULT_EXTRACTION, DEFAULT_TRANSLATION, DEFAULT_PDF_GENERATION, OPENROUTER_API_KEY
from webapp.database import get_db, init_db
from webapp.models import (
    create_job, get_job, list_jobs, update_job_status,
    create_document, list_documents, delete_document, get_document,
    create_user, get_user_by_username, get_user_by_email, update_user_settings,
)
from webapp.tasks import start_job, get_job_status
from webapp.auth import (
    hash_password, verify_password, set_auth_cookie, clear_auth_cookie,
    require_user, optional_user, User,
)

app = FastAPI(title="ReadingBuddy")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(optional_user)):
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User | None = Depends(optional_user)):
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html", context={"error": request.query_params.get("error")})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    response = RedirectResponse(url="/dashboard", status_code=303)
    set_auth_cookie(response, user.id)
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User | None = Depends(optional_user)):
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(request, "register.html", context={"error": request.query_params.get("error")})


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != confirm_password:
        return RedirectResponse(url="/register?error=password_mismatch", status_code=303)
    if len(password) < 6:
        return RedirectResponse(url="/register?error=password_short", status_code=303)
    if get_user_by_username(db, username):
        return RedirectResponse(url="/register?error=username_taken", status_code=303)
    if get_user_by_email(db, email):
        return RedirectResponse(url="/register?error=email_taken", status_code=303)
    create_user(db, username=username, email=email, hashed_password=hash_password(password))
    response = RedirectResponse(url="/dashboard", status_code=303)
    new_user = get_user_by_username(db, username)
    set_auth_cookie(response, new_user.id)
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    clear_auth_cookie(response)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    jobs = list_jobs(db, user.id, limit=20)
    documents = list_documents(db, user.id, limit=20)
    return templates.TemplateResponse(request, "dashboard.html", context={
        "user": user,
        "jobs": jobs,
        "documents": documents,
        "has_backend_key": bool(OPENROUTER_API_KEY),
    })


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    documents = list_documents(db, user.id, limit=50)
    return templates.TemplateResponse(request, "documents.html", context={"user": user, "documents": documents})


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    upload_dir = WORKSPACE / "users" / user.id / "documents"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".pdf"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = upload_dir / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    doc = create_document(db, user_id=user.id, filename=saved_name, original_name=file.filename, file_size=len(content))
    return RedirectResponse(url="/documents", status_code=303)


@app.post("/documents/{doc_id}/delete")
async def delete_user_document(
    doc_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    delete_document(db, doc_id, user.id)
    return RedirectResponse(url="/documents", status_code=303)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    dpi: int = Form(300),
    ocr_mode: str = Form("auto"),
    lang: str = Form("eng"),
    model: str = Form("openai/gpt-4o-mini"),
    temperature: float = Form(0.2),
    page_size: str = Form("A4"),
    margin: str = Form("18mm"),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    api_key = user.openrouter_api_key or OPENROUTER_API_KEY
    if not api_key:
        return RedirectResponse(url="/settings?error=no_api_key", status_code=303)

    upload_dir = WORKSPACE / "users" / user.id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".pdf"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = upload_dir / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    doc = create_document(db, user_id=user.id, filename=saved_name, original_name=file.filename, file_size=len(content))

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

    job = create_job(db, user_id=user.id, filename=file.filename, settings=settings, document_id=doc.id)
    start_job(job.id, str(saved_path), settings)

    return RedirectResponse(url=f"/job/{job.id}", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    jobs = list_jobs(db, user.id, limit=50)
    return templates.TemplateResponse(request, "jobs.html", context={"user": user, "jobs": jobs})


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = get_job(db, job_id, user.id)
    if not job:
        return HTMLResponse("Job not found", status_code=404)
    return templates.TemplateResponse(request, "job_detail.html", context={"user": user, "job": job})


@app.get("/job/{job_id}/status")
async def job_status(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = get_job(db, job_id, user.id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
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
async def download_pdf(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = get_job(db, job_id, user.id)
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
async def download_markdown(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = get_job(db, job_id, user.id)
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
async def cancel_job(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = get_job(db, job_id, user.id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    if job.status in ("pending", "processing"):
        update_job_status(db, job_id, "cancelled", error="Cancelled by user")
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(require_user)):
    error = request.query_params.get("error")
    success = request.query_params.get("success")
    return templates.TemplateResponse(request, "settings.html", context={
        "user": user,
        "error": error,
        "success": success,
    })


@app.post("/settings")
async def update_settings(
    request: Request,
    openrouter_api_key: str = Form(""),
    openrouter_model: str = Form("openai/gpt-4o-mini"),
    extraction_dpi: int = Form(300),
    extraction_ocr_mode: str = Form("auto"),
    extraction_lang: str = Form("eng"),
    translation_temperature: float = Form(0.2),
    pdf_page_size: str = Form("A4"),
    pdf_margin: str = Form("18mm"),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    update_user_settings(db, user.id, {
        "openrouter_api_key": openrouter_api_key,
        "openrouter_model": openrouter_model,
        "extraction_dpi": extraction_dpi,
        "extraction_ocr_mode": extraction_ocr_mode,
        "extraction_lang": extraction_lang,
        "translation_temperature": translation_temperature,
        "pdf_page_size": pdf_page_size,
        "pdf_margin": pdf_margin,
    })
    return RedirectResponse(url="/settings?success=saved", status_code=303)


@app.get("/health")
async def health():
    return {"status": "ok"}
