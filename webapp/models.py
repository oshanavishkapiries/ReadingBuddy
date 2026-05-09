import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON
from sqlalchemy.orm import Session

from webapp.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
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
    page_count = Column(Integer, default=0)

    @property
    def workspace_dir(self):
        return f"workspace/{self.id}"


def create_job(db: Session, filename: str, settings: dict) -> Job:
    job = Job(filename=filename, settings=settings, status="pending", current_step="Queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Job | None:
    return db.query(Job).filter(Job.id == job_id).first()


def list_jobs(db: Session, limit: int = 20) -> list[Job]:
    return db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()


def update_job_progress(db: Session, job_id: str, progress: float, step: str, detail: str = ""):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.progress = progress
        job.current_step = step
        job.step_detail = detail
        job.updated_at = datetime.utcnow()
        db.commit()


def update_job_status(db: Session, job_id: str, status: str, error: str = "", output_pdf: str = ""):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.error = error
        job.output_pdf = output_pdf
        job.updated_at = datetime.utcnow()
        db.commit()
