import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.ratelimit import limit_signup
from app.db.session import get_db
from app.search.schema import JobOut, ProfileSignals
from app.search.service import job_to_out
from app.services import users as svc

router = APIRouter(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    email: str | None = Field(None, max_length=320)


class ProfilePatch(BaseModel):
    """Lists are merged into the stored profile; scalars replace it."""

    summary: str | None = None
    skills: list[str] | None = None
    remove_skills: list[str] | None = None
    target_roles: list[str] | None = None
    preferred_locations: list[str] | None = None
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] | None = None
    work_mode: Literal["remote", "hybrid", "onsite"] | None = None
    job_type: Literal["full_time", "internship", "contract", "part_time"] | None = None
    min_salary: int | None = Field(None, ge=0)


class FeedbackIn(BaseModel):
    job_id: int
    action: Literal["saved", "dismissed", "clicked"]


def _guard(fn):
    try:
        return fn()
    except svc.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("", status_code=201)
def create_user(body: UserCreate, request: Request, db: Session = Depends(get_db)) -> dict:
    limit_signup(request)
    user = svc.create_user(db, body.email)
    db.commit()
    return {"id": user.id}


@router.get("/{user_id}/profile", response_model=ProfileSignals)
def read_profile(user_id: uuid.UUID, db: Session = Depends(get_db)):
    return svc.profile_signals(_guard(lambda: svc.get_profile(db, user_id)))


@router.patch("/{user_id}/profile", response_model=ProfileSignals)
def patch_profile(user_id: uuid.UUID, body: ProfilePatch, db: Session = Depends(get_db)):
    changes = body.model_dump(exclude_unset=True)
    profile = _guard(lambda: svc.update_profile(db, user_id, changes))
    db.commit()
    return svc.profile_signals(profile)


@router.delete("/{user_id}/profile", response_model=ProfileSignals)
def reset_profile(user_id: uuid.UUID, db: Session = Depends(get_db)):
    """'Forget what you know about me.'"""
    profile = _guard(lambda: svc.clear_profile(db, user_id))
    db.commit()
    return svc.profile_signals(profile)


@router.post("/{user_id}/feedback", status_code=204)
def add_feedback(user_id: uuid.UUID, body: FeedbackIn, db: Session = Depends(get_db)) -> None:
    _guard(lambda: svc.record_feedback(db, user_id, body.job_id, body.action))
    db.commit()


@router.delete("/{user_id}/feedback/{action}/{job_id}", status_code=204)
def delete_feedback(user_id: uuid.UUID, action: str, job_id: int, db: Session = Depends(get_db)) -> None:
    _guard(lambda: svc.get_user(db, user_id))
    svc.remove_feedback(db, user_id, job_id, action)
    db.commit()


@router.get("/{user_id}/saved", response_model=list[JobOut])
def saved_jobs(user_id: uuid.UUID, db: Session = Depends(get_db)):
    _guard(lambda: svc.get_user(db, user_id))
    return [job_to_out(j) for j in svc.feedback_jobs(db, user_id, "saved")]
