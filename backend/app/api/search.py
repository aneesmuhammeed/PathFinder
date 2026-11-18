from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_embedder_dep
from app.db.models import Job
from app.db.session import get_db
from app.search.embedder import Embedder
from app.search.schema import JobOut, SearchRequest, SearchResponse
from app.search.service import InvalidSearch, job_to_out, search_jobs
from app.services.users import NotFound

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_db), embedder: Embedder = Depends(get_embedder_dep)):
    """Hybrid (semantic + keyword) search with optional personalised, explainable re-ranking."""
    try:
        return search_jobs(db, embedder, req)
    except InvalidSearch as exc:
        raise HTTPException(422, str(exc)) from exc
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id, options=[selectinload(Job.company)])
    if job is None:
        raise HTTPException(404, "job not found")
    return job_to_out(job)
