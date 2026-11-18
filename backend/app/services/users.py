"""Users, long-term profile memory and job feedback. Shared by the REST API and the agent's tools."""

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import FEEDBACK_ACTIONS, Job, JobFeedback, User, UserProfile
from app.ingest.skills import canonicalize
from app.search.schema import ProfileSignals

LIST_FIELDS = ("skills", "target_roles", "preferred_locations")
SCALAR_FIELDS = ("summary", "seniority", "work_mode", "job_type", "min_salary")
# Saving a job cancels a dismissal and vice versa; "clicked" is independent.
OPPOSITE = {"saved": "dismissed", "dismissed": "saved"}


class NotFound(LookupError):
    pass


def create_user(session: Session, email: str | None = None) -> User:
    user = User(email=email)
    session.add(user)
    session.flush()
    session.add(UserProfile(user_id=user.id))
    session.flush()
    return user


def get_user(session: Session, user_id: uuid.UUID) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise NotFound(f"user {user_id} not found")
    return user


def get_profile(session: Session, user_id: uuid.UUID) -> UserProfile:
    get_user(session, user_id)
    profile = session.get(UserProfile, user_id)
    if profile is None:  # users created outside create_user()
        profile = UserProfile(user_id=user_id)
        session.add(profile)
        session.flush()
    return profile


def update_profile(session: Session, user_id: uuid.UUID, changes: dict[str, Any]) -> UserProfile:
    """Merge semantics: list fields are unioned (the user said they also know SQL; we don't forget
    Python), scalar fields are replaced. Pass `None` for a scalar to clear it. Unknown keys are rejected
    so an LLM-generated tool call with a typo fails loudly instead of silently doing nothing."""
    unknown = set(changes) - set(LIST_FIELDS) - set(SCALAR_FIELDS) - {"remove_skills"}
    if unknown:
        raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    p = get_profile(session, user_id)
    for f in LIST_FIELDS:
        if changes.get(f):
            new = [canonicalize(x) if f == "skills" else x.strip() for x in changes[f]]
            setattr(p, f, list(dict.fromkeys([*getattr(p, f), *[x for x in new if x]])))
    if changes.get("remove_skills"):
        gone = {canonicalize(x) for x in changes["remove_skills"]}
        p.skills = [s for s in p.skills if s not in gone]
    for f in SCALAR_FIELDS:
        if f in changes:
            setattr(p, f, changes[f])
    session.flush()
    return p


def clear_profile(session: Session, user_id: uuid.UUID) -> UserProfile:
    """Forget everything saved about the user (they can always ask the agent to remember again)."""
    p = get_profile(session, user_id)
    p.summary = p.seniority = p.work_mode = p.job_type = p.min_salary = None
    p.skills, p.target_roles, p.preferred_locations, p.facts = [], [], [], {}
    session.flush()
    return p


def profile_signals(profile: UserProfile | None) -> ProfileSignals | None:
    if profile is None:
        return None
    return ProfileSignals(
        skills=profile.skills or [], target_roles=profile.target_roles or [],
        preferred_locations=profile.preferred_locations or [], seniority=profile.seniority,
        work_mode=profile.work_mode, job_type=profile.job_type, min_salary=profile.min_salary,
    )


def record_feedback(session: Session, user_id: uuid.UUID, job_id: int, action: str) -> None:
    if action not in FEEDBACK_ACTIONS:
        raise ValueError(f"action must be one of {FEEDBACK_ACTIONS}")
    get_user(session, user_id)
    if session.get(Job, job_id) is None:
        raise NotFound(f"job {job_id} not found")
    if action in OPPOSITE:
        session.execute(delete(JobFeedback).where(
            JobFeedback.user_id == user_id, JobFeedback.job_id == job_id, JobFeedback.action == OPPOSITE[action]))
    exists = session.scalar(select(JobFeedback.id).where(
        JobFeedback.user_id == user_id, JobFeedback.job_id == job_id, JobFeedback.action == action))
    if exists is None:
        session.add(JobFeedback(user_id=user_id, job_id=job_id, action=action))
    session.flush()


def remove_feedback(session: Session, user_id: uuid.UUID, job_id: int, action: str) -> None:
    session.execute(delete(JobFeedback).where(
        JobFeedback.user_id == user_id, JobFeedback.job_id == job_id, JobFeedback.action == action))
    session.flush()


def feedback_jobs(session: Session, user_id: uuid.UUID, action: str) -> list[Job]:
    stmt = (select(Job).join(JobFeedback, JobFeedback.job_id == Job.id)
            .where(JobFeedback.user_id == user_id, JobFeedback.action == action)
            .order_by(JobFeedback.created_at.desc(), JobFeedback.id.desc()))
    return list(session.scalars(stmt))
