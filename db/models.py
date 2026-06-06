"""
db/models.py
────────────
SQLAlchemy ORM models for PostgreSQL.
Tables: users, sessions, conversations, messages, citations, agent_runs, research_jobs, query_metrics
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, JSON, text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_active = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    metadata_ = Column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_last_active", "last_active"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    session = relationship("Session", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_conversations_session_id", "session_id"),)


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(64), nullable=False)  # denorm for fast lookup
    role = Column(String(20), nullable=False)         # user | assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship("CitationRecord", back_populates="message", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("ix_messages_conversation_id", "conversation_id"),
    )


class CitationRecord(Base):
    __tablename__ = "citations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    title = Column(Text, nullable=False, default="")
    url = Column(Text, nullable=False)
    snippet = Column(Text, default="")
    source_type = Column(String(20), default="web")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    message = relationship("Message", back_populates="citations")

    __table_args__ = (Index("ix_citations_message_id", "message_id"),)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(64), unique=True, nullable=False)
    session_id = Column(String(64), nullable=True)
    query = Column(Text, nullable=False)
    complexity = Column(String(20), default="MEDIUM")  # SIMPLE | MEDIUM | COMPLEX | RESEARCH
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    cache_hit = Column(Boolean, default=False)
    spans = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_agent_runs_run_id", "run_id"),
        Index("ix_agent_runs_session_id", "session_id"),
        Index("ix_agent_runs_created_at", "created_at"),
    )


class ResearchJob(Base):
    __tablename__ = "research_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(64), unique=True, nullable=False)
    query = Column(Text, nullable=False)
    status = Column(String(20), default="pending")  # pending | running | done | failed
    progress = Column(Integer, default=0)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_research_jobs_job_id", "job_id"),
        Index("ix_research_jobs_status", "status"),
    )


class QueryMetric(Base):
    __tablename__ = "query_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(64), nullable=False)
    complexity = Column(String(20), nullable=False)
    latency_ms = Column(Float, nullable=False)
    tokens = Column(Integer, default=0)
    cache_hit = Column(Boolean, default=False)
    search_latency_ms = Column(Float, default=0.0)
    rerank_latency_ms = Column(Float, default=0.0)
    llm_latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_query_metrics_created_at", "created_at"),
        Index("ix_query_metrics_complexity", "complexity"),
    )
