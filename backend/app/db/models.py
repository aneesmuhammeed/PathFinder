"""Relational schema.

Design notes
------------
* ``jobs`` is the hot table. ``(source, external_id)`` is the natural key, so ingestion is an
  idempotent upsert. ``content_hash`` lets us skip unchanged rows and only re-embed what changed.
* Jobs are never deleted: when a posting disappears from its source it is *expired*
  (``expired_at``) so saved jobs / feedback / conversation history keep working.
* ``search_tsv`` is a STORED generated column (title weighted above description) backing the
  full-text half of hybrid search; ``embedding`` backs the vector half (HNSW, cosine).
* Skills are normalised (``skills`` + ``job_skills``) so we can filter and score on overlap in SQL.
* Messages store content as JSONB blocks so tool calls / tool results round-trip losslessly.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim

SENIORITY = ("intern", "junior", "mid", "senior", "lead")
JOB_TYPES = ("full_time", "internship", "contract", "part_time")
WORK_MODES = ("remote", "hybrid", "onsite")
FEEDBACK_ACTIONS = ("saved", "dismissed", "clicked")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    website: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship(back_populates="company")


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)  # canonical, lower-case
    category: Mapped[str | None] = mapped_column(String(50))


class JobSkill(Base):
    __tablename__ = "job_skills"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)

    __table_args__ = (Index("ix_job_skills_skill_id", "skill_id"),)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    external_id: Mapped[str] = mapped_column(String(200))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))

    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1000))

    location: Mapped[str | None] = mapped_column(String(200))
    country: Mapped[str | None] = mapped_column(String(100))
    work_mode: Mapped[str] = mapped_column(String(20), default="onsite")
    job_type: Mapped[str] = mapped_column(String(20), default="full_time")
    seniority: Mapped[str] = mapped_column(String(20), default="mid")

    salary_min: Mapped[int | None] = mapped_column(Integer)
    salary_max: Mapped[int | None] = mapped_column(Integer)
    salary_currency: Mapped[str | None] = mapped_column(String(3))
    salary_period: Mapped[str | None] = mapped_column(String(10))  # 'year' | 'month'

    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedded_hash: Mapped[str | None] = mapped_column(String(64))  # content_hash the vector was built from

    search_tsv = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(description, '')), 'B')",
            persisted=True,
        ),
    )

    company: Mapped[Company] = relationship(back_populates="jobs")
    skills: Mapped[list[Skill]] = relationship(secondary="job_skills", lazy="selectin")

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_jobs_source_external_id"),
        CheckConstraint(_in("seniority", SENIORITY), name="ck_jobs_seniority"),
        CheckConstraint(_in("job_type", JOB_TYPES), name="ck_jobs_job_type"),
        CheckConstraint(_in("work_mode", WORK_MODES), name="ck_jobs_work_mode"),
        Index("ix_jobs_search_tsv", "search_tsv", postgresql_using="gin"),
        Index(
            "ix_jobs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Most queries only look at live jobs.
        Index("ix_jobs_active_posted", "posted_at", postgresql_where=text("expired_at IS NULL")),
        Index("ix_jobs_company_id", "company_id"),
    )


class User(Base):
    """Anonymous-first: a user is just a UUID the browser keeps; email is optional."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    profile: Mapped["UserProfile | None"] = relationship(back_populates="user", uselist=False)


class UserProfile(Base):
    """Long-term memory the agent builds up about the user across conversations."""

    __tablename__ = "user_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    target_roles: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    preferred_locations: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    seniority: Mapped[str | None] = mapped_column(String(20))
    work_mode: Mapped[str | None] = mapped_column(String(20))
    job_type: Mapped[str | None] = mapped_column(String(20))
    min_salary: Mapped[int | None] = mapped_column(Integer)
    facts: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")  # free-form notes
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="profile")

    __table_args__ = (
        CheckConstraint("seniority IS NULL OR " + _in("seniority", SENIORITY), name="ck_profiles_seniority"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(200))
    summary: Mapped[str | None] = mapped_column(Text)  # rolling summary of trimmed history
    summarized_through_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.id", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[list] = mapped_column(JSONB)  # list of content blocks (text / tool_use / tool_result)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint(_in("role", ("user", "assistant")), name="ck_messages_role"),
        Index("ix_messages_conversation_id_id", "conversation_id", "id"),
    )


class JobFeedback(Base):
    """Explicit + implicit feedback; the personalisation signal for the re-ranker."""

    __tablename__ = "job_feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    action: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship()

    __table_args__ = (
        UniqueConstraint("user_id", "job_id", "action", name="uq_feedback_user_job_action"),
        CheckConstraint(_in("action", FEEDBACK_ACTIONS), name="ck_feedback_action"),
        Index("ix_feedback_user_action", "user_id", "action"),
    )


class IngestionRun(Base):
    """Audit trail for every ingestion run (what changed, how long, did it fail)."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | success | failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched: Mapped[int] = mapped_column(Integer, default=0)
    inserted: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    unchanged: Mapped[int] = mapped_column(Integer, default=0)
    expired: Mapped[int] = mapped_column(Integer, default=0)
    embedded: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(_in("status", ("running", "success", "failed")), name="ck_ingestion_status"),
        Index("ix_ingestion_runs_source_started", "source", "started_at"),
    )
