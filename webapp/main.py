import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from webapp.config import WORKSPACE, DEFAULT_EXTRACTION, DEFAULT_TRANSLATION, DEFAULT_PDF_GENERATION, OPENROUTER_API_KEY
from webapp.database import get_db, init_db
from webapp.models import (
    create_job, get_job, list_jobs, update_job_status,
    create_document, list_documents, delete_document, get_document,
    create_user, get_user_by_username, get_user_by_email, get_user_by_id, update_user_settings,
    create_shared_document, get_shared_document, get_shared_by_job,
    delete_shared_document, list_shared_documents, like_shared_document,
    get_today_usage, log_usage,
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


def flash(type: str, message: str, duration: int = 4000) -> dict:
    return {"type": type, "message": message, "duration": duration}


class NotifyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        notify_cookie = request.cookies.get("rb_notify", "")
        if notify_cookie and response.headers.get("content-type", "").startswith("text/html"):
            parts = notify_cookie.split("|", 2)
            if len(parts) == 3:
                notify_html = f'<script src="/static/js/notify.js"></script><script>document.addEventListener("DOMContentLoaded",()=>{{Notify.{parts[0]}("{parts[1]}",{parts[2]})}})</script>'
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                body = body.replace(b"</body>", notify_html.encode() + b"</body>")
                response.headers["content-length"] = str(len(body))
                async def iter_body():
                    yield body
                response.body_iterator = iter_body()
        if notify_cookie:
            response.delete_cookie(key="rb_notify", path="/")
        return response


app.add_middleware(NotifyMiddleware)


@app.on_event("startup")
def startup():
    init_db()
    from webapp.config import OPENROUTER_API_KEY
    if OPENROUTER_API_KEY:
        print(f"Backend OpenRouter API key loaded (key: ...{OPENROUTER_API_KEY[-4:]})")
    else:
        print("WARNING: No backend OpenRouter API key found in environment")


def _save_local_file(content: bytes, filename: str, base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / filename
    path.write_bytes(content)
    return str(path)


def _save_local_temp(content: bytes, filename: str) -> str:
    tmp_dir = WORKSPACE / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / filename
    path.write_bytes(content)
    return str(path)


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
    response.set_cookie(
        key="rb_notify",
        value="success|Welcome to ReadingBuddy, " + username + "!|5000",
        max_age=5,
        path="/",
    )
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    clear_auth_cookie(response)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    jobs = list_jobs(db, user.id, limit=20)
    active_jobs = [j for j in jobs if j.status in ("pending", "processing")]
    documents = list_documents(db, user.id, limit=20)
    usage_count, _ = get_today_usage(db, user.id) if not user.openrouter_api_key else (0, 0)
    return templates.TemplateResponse(request, "dashboard.html", context={
        "user": user,
        "jobs": active_jobs if active_jobs else jobs[:5],
        "all_jobs": jobs,
        "documents": documents,
        "has_backend_key": bool(OPENROUTER_API_KEY),
        "usage_count": usage_count,
    })


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    documents = list_documents(db, user.id, limit=50)
    usage_count, _ = get_today_usage(db, user.id) if not user.openrouter_api_key else (0, 0)
    return templates.TemplateResponse(request, "documents.html", context={
        "user": user,
        "documents": documents,
        "usage_count": usage_count,
        "has_backend_key": bool(OPENROUTER_API_KEY),
    })


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
    create_document(db, user_id=user.id, filename=saved_name, original_name=file.filename, file_size=len(content))
    response = RedirectResponse(url="/documents", status_code=303)
    response.set_cookie(key="rb_notify", value="success|Document uploaded successfully|3000", max_age=5, path="/")
    return response


@app.post("/documents/{doc_id}/delete")
async def delete_user_document(
    doc_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    delete_document(db, doc_id, user.id)
    response = RedirectResponse(url="/documents", status_code=303)
    response.set_cookie(key="rb_notify", value="success|Document deleted|3000", max_age=5, path="/")
    return response


@app.post("/documents/{doc_id}/translate")
async def translate_document(
    doc_id: str,
    dpi: int = Form(300),
    ocr_mode: str = Form("auto"),
    lang: str = Form("eng"),
    model: str = Form("openai/gpt-4o-mini"),
    temperature: float = Form(0.2),
    page_size: str = Form("A4"),
    margin: str = Form("18mm"),
    font_size: float = Form(16.5),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    api_key = user.openrouter_api_key or OPENROUTER_API_KEY
    if not api_key:
        response = RedirectResponse(url="/settings", status_code=303)
        response.set_cookie(key="rb_notify", value="error|No API key configured. Add one in Settings.|6000", max_age=5, path="/")
        return response

    if not user.openrouter_api_key and OPENROUTER_API_KEY:
        usage_count, _ = get_today_usage(db, user.id)
        if usage_count >= 3:
            response = RedirectResponse(url="/documents", status_code=303)
            response.set_cookie(key="rb_notify", value="warning|Daily limit reached (3/3). Add your own API key for unlimited access.|6000", max_age=5, path="/")
            return response

    doc = get_document(db, doc_id, user.id)
    if not doc:
        return RedirectResponse(url="/documents", status_code=303)

    upload_dir = WORKSPACE / "users" / user.id / "documents"
    local_path = str(upload_dir / doc.filename)
    if not Path(local_path).exists():
        return RedirectResponse(url="/documents", status_code=303)

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
            "font_size": font_size,
        },
        "_user_id": user.id,
    }

    job = create_job(db, user_id=user.id, filename=doc.original_name, settings=settings, document_id=doc.id)
    if not user.openrouter_api_key:
        log_usage(db, user.id, job.id, 0)
    start_job(job.id, local_path, settings)

    return RedirectResponse(url=f"/job/{job.id}", status_code=303)


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
    font_size: float = Form(16.5),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    api_key = user.openrouter_api_key or OPENROUTER_API_KEY
    if not api_key:
        response = RedirectResponse(url="/settings", status_code=303)
        response.set_cookie(key="rb_notify", value="error|No API key configured. Add one in Settings.|6000", max_age=5, path="/")
        return response

    if not user.openrouter_api_key and OPENROUTER_API_KEY:
        usage_count, _ = get_today_usage(db, user.id)
        if usage_count >= 3:
            response = RedirectResponse(url="/dashboard", status_code=303)
            response.set_cookie(key="rb_notify", value="warning|Daily limit reached (3/3). Add your own API key for unlimited access.|6000", max_age=5, path="/")
            return response

    content = await file.read()
    ext = Path(file.filename).suffix or ".pdf"
    saved_name = f"{uuid.uuid4().hex}{ext}"

    upload_dir = WORKSPACE / "users" / user.id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    local_path = str(upload_dir / saved_name)
    Path(local_path).write_bytes(content)

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
            "font_size": font_size,
        },
        "_user_id": user.id,
    }

    job = create_job(db, user_id=user.id, filename=file.filename, settings=settings, document_id=doc.id)
    if not user.openrouter_api_key:
        log_usage(db, user.id, job.id, 0)
    start_job(job.id, local_path, settings)

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
    shared = get_shared_by_job(db, job_id)
    return templates.TemplateResponse(request, "job_detail.html", context={"user": user, "job": job, "shared": shared})


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
    if not job:
        return HTMLResponse("PDF not available", status_code=404)
    if job.output_pdf:
        pdf_path = Path(job.output_pdf)
        if pdf_path.exists():
            return FileResponse(
                path=str(pdf_path),
                filename=f"readingbuddy_{job_id}.pdf",
                media_type="application/pdf",
            )
    return HTMLResponse("PDF not available", status_code=404)


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


@app.post("/job/{job_id}/share")
async def share_job(
    job_id: str,
    public_name: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = get_job(db, job_id, user.id)
    if not job or job.status != "completed":
        return RedirectResponse(url=f"/job/{job_id}", status_code=303)
    existing = get_shared_by_job(db, job_id)
    if existing:
        return RedirectResponse(url=f"/job/{job_id}", status_code=303)

    name = public_name.strip() or job.filename
    create_shared_document(db, job_id=job_id, user_id=user.id, public_name=name)

    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.post("/job/{job_id}/unshare")
async def unshare_job(
    job_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    shared = get_shared_by_job(db, job_id)
    if shared and shared.user_id == user.id:
        delete_shared_document(db, shared.id, user.id)
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.post("/shared/{shared_id}/like")
async def like_document(
    shared_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    like_shared_document(db, shared_id)
    referer = request.headers.get("referer", "/explore")
    return RedirectResponse(url=referer, status_code=303)


@app.get("/explore", response_class=HTMLResponse)
async def explore_page(request: Request, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    shared_docs = list_shared_documents(db, limit=50)
    return templates.TemplateResponse(request, "explore.html", context={
        "user": user,
        "shared_docs": shared_docs,
    })


@app.get("/shared/{shared_id}", response_class=HTMLResponse)
async def view_shared_document(shared_id: str, request: Request, db: Session = Depends(get_db), user: User | None = Depends(optional_user)):
    shared = get_shared_document(db, shared_id)
    if not shared:
        return HTMLResponse("Document not found", status_code=404)
    job = shared.job
    return templates.TemplateResponse(request, "shared_view.html", context={
        "user": user,
        "shared": shared,
        "job": job,
    })


@app.get("/shared/{shared_id}/download")
async def download_shared(shared_id: str, db: Session = Depends(get_db)):
    shared = get_shared_document(db, shared_id)
    if not shared:
        return HTMLResponse("Document not found", status_code=404)
    if shared.job and shared.job.output_pdf:
        pdf_path = Path(shared.job.output_pdf)
        if pdf_path.exists():
            return FileResponse(
                path=str(pdf_path),
                filename=f"readingbuddy_{shared.public_name}.pdf",
                media_type="application/pdf",
            )
    return HTMLResponse("Download not available", status_code=404)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "settings.html", context={"user": user})


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
    pdf_font_size: float = Form(16.5),
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
        "pdf_font_size": pdf_font_size,
    })
    response = RedirectResponse(url="/settings", status_code=303)
    response.set_cookie(key="rb_notify", value="success|Settings saved successfully|3000", max_age=5, path="/")
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "storage": "local"}


def _get_user_from_request(request: Request) -> User | None:
    try:
        from webapp.auth import decode_access_token
        from webapp.database import get_db
        token = request.cookies.get("rb_session")
        if not token:
            return None
        payload = decode_access_token(token)
        if not payload:
            return None
        user_id = payload.get("sub")
        if not user_id:
            return None
        db_gen = get_db()
        db = next(db_gen)
        try:
            return get_user_by_id(db, user_id)
        finally:
            db.close()
    except Exception:
        return None


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    user = _get_user_from_request(request)
    return templates.TemplateResponse(request, "404.html", context={"user": user}, status_code=404)


@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    return templates.TemplateResponse(request, "401.html", context={"user": None}, status_code=401)


@app.exception_handler(403)
async def forbidden_handler(request: Request, exc):
    user = _get_user_from_request(request)
    return templates.TemplateResponse(request, "403.html", context={"user": user}, status_code=403)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    user = _get_user_from_request(request)
    return templates.TemplateResponse(request, "500.html", context={"user": user}, status_code=500)
