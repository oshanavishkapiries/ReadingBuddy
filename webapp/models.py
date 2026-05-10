import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import Session, relationship

from webapp.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    openrouter_api_key = Column(String, default="")
    openrouter_model = Column(String, default="openai/gpt-4o-mini")
    extraction_dpi = Column(Integer, default=300)
    extraction_ocr_mode = Column(String, default="auto")
    extraction_lang = Column(String, default="eng")
    translation_temperature = Column(Float, default=0.2)
    pdf_page_size = Column(String, default="A4")
    pdf_margin = Column(String, default="18mm")
    pdf_font_size = Column(Float, default=16.5)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    drive_file_id = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")
    jobs = relationship("Job", back_populates="document")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    current_step = Column(String, default="")
    step_detail = Column(Text, default="")
    error = Column(Text, default="")
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    output_pdf = Column(String, default="")
    output_pdf_drive_id = Column(String, default="")
    page_count = Column(Integer, default=0)

    user = relationship("User", back_populates="jobs")
    document = relationship("Document", back_populates="jobs")

    @property
    def workspace_dir(self):
        return f"workspace/{self.id}"


def create_user(db: Session, username: str, email: str, hashed_password: str) -> User:
    user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def update_user_settings(db: Session, user_id: str, settings: dict) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        for key, value in settings.items():
            if hasattr(user, key):
                setattr(user, key, value)
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
    return user


def create_document(db: Session, user_id: str, filename: str, original_name: str, file_size: int = 0, drive_file_id: str = "") -> Document:
    doc = Document(user_id=user_id, filename=filename, original_name=original_name, file_size=file_size, drive_file_id=drive_file_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, doc_id: str, user_id: str | None = None) -> Document | None:
    q = db.query(Document).filter(Document.id == doc_id)
    if user_id:
        q = q.filter(Document.user_id == user_id)
    return q.first()


def list_documents(db: Session, user_id: str, limit: int = 50) -> list[Document]:
    return db.query(Document).filter(Document.user_id == user_id).order_by(Document.created_at.desc()).limit(limit).all()


def delete_document(db: Session, doc_id: str, user_id: str) -> bool:
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == user_id).first()
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False


def create_job(db: Session, user_id: str, filename: str, settings: dict, document_id: str | None = None) -> Job:
    job = Job(user_id=user_id, document_id=document_id, filename=filename, settings=settings, status="pending", current_step="Queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str, user_id: str | None = None) -> Job | None:
    q = db.query(Job).filter(Job.id == job_id)
    if user_id:
        q = q.filter(Job.user_id == user_id)
    return q.first()


def list_jobs(db: Session, user_id: str, limit: int = 50) -> list[Job]:
    return db.query(Job).filter(Job.user_id == user_id).order_by(Job.created_at.desc()).limit(limit).all()


def update_job_progress(db: Session, job_id: str, progress: float, step: str, detail: str = ""):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.progress = progress
        job.current_step = step
        job.step_detail = detail
        job.updated_at = datetime.utcnow()
        db.commit()


class SharedDocument(Base):
    __tablename__ = "shared_documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True, unique=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    public_name = Column(String, nullable=False)
    drive_file_id = Column(String, default="")
    direct_link = Column(String, default="")
    likes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job")
    user = relationship("User")


def create_shared_document(db: Session, job_id: str, user_id: str, public_name: str) -> SharedDocument:
    shared = SharedDocument(job_id=job_id, user_id=user_id, public_name=public_name)
    db.add(shared)
    db.commit()
    db.refresh(shared)
    return shared


def get_shared_document(db: Session, shared_id: str) -> SharedDocument | None:
    return db.query(SharedDocument).filter(SharedDocument.id == shared_id).first()


def get_shared_by_job(db: Session, job_id: str) -> SharedDocument | None:
    return db.query(SharedDocument).filter(SharedDocument.job_id == job_id).first()


def delete_shared_document(db: Session, shared_id: str, user_id: str) -> bool:
    shared = db.query(SharedDocument).filter(SharedDocument.id == shared_id, SharedDocument.user_id == user_id).first()
    if shared:
        db.delete(shared)
        db.commit()
        return True
    return False


def list_shared_documents(db: Session, limit: int = 50) -> list[SharedDocument]:
    return db.query(SharedDocument).order_by(SharedDocument.created_at.desc()).limit(limit).all()


def like_shared_document(db: Session, shared_id: str) -> SharedDocument | None:
    shared = db.query(SharedDocument).filter(SharedDocument.id == shared_id).first()
    if shared:
        shared.likes += 1
        db.commit()
        db.refresh(shared)
    return shared


def update_job_status(db: Session, job_id: str, status: str, error: str = "", output_pdf: str = "", output_pdf_drive_id: str = ""):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.error = error
        job.output_pdf = output_pdf
        if output_pdf_drive_id:
            job.output_pdf_drive_id = output_pdf_drive_id
        if status == "completed":
            job.page_count = job.settings.get("page_count", job.page_count)
        job.updated_at = datetime.utcnow()
        db.commit()


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    date = Column(String, nullable=False, index=True)
    page_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


def get_today_usage(db: Session, user_id: str) -> tuple[int, int]:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    logs = db.query(UsageLog).filter(UsageLog.user_id == user_id, UsageLog.date == today).all()
    count = len(logs)
    total_pages = sum(log.page_count for log in logs)
    return count, total_pages


def log_usage(db: Session, user_id: str, job_id: str, page_count: int):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    entry = UsageLog(user_id=user_id, job_id=job_id, date=today, page_count=page_count)
    db.add(entry)
    db.commit()
    return entry
