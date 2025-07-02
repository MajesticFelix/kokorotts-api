"""Database models for API key management."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Project(Base):
    """Project model for organizing API keys."""
    
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    
    # Relationship to API keys
    api_keys = relationship("APIKey", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"


class APIKey(Base):
    """API Key model for authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(50), nullable=False, index=True)  # e.g., "kk_mobile_app"
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    
    # Metadata
    description = Column(Text, nullable=True)
    rate_limit_tier = Column(String(50), default="standard")
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    
    # Relationship to project
    project = relationship("Project", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, prefix='{self.key_prefix}', active={self.is_active})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if the API key is valid (active and not expired)."""
        return self.is_active and not self.is_expired
    
    def update_usage(self, character_count: int = 0) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.total_characters += character_count
        self.last_used_at = datetime.utcnow()


class RateLimitTier:
    """Rate limit tier definitions."""
    
    TIERS = {
        "free": {
            "requests_per_minute": 10,
            "requests_per_hour": 100,
            "requests_per_day": 1000,
            "characters_per_request": 1000,
            "characters_per_hour": 10000,
            "characters_per_day": 50000,
            "max_concurrent_requests": 1
        },
        "standard": {
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "characters_per_request": 10000,
            "characters_per_hour": 100000,
            "characters_per_day": 500000,
            "max_concurrent_requests": 5
        },
        "premium": {
            "requests_per_minute": 500,
            "requests_per_hour": 5000,
            "requests_per_day": 50000,
            "characters_per_request": 50000,
            "characters_per_hour": 500000,
            "characters_per_day": 2000000,
            "max_concurrent_requests": 10
        },
        "unlimited": {
            "requests_per_minute": -1,  # No limit
            "requests_per_hour": -1,
            "requests_per_day": -1,
            "characters_per_request": -1,
            "characters_per_hour": -1,
            "characters_per_day": -1,
            "max_concurrent_requests": -1
        }
    }
    
    @classmethod
    def get_limits(cls, tier: str) -> dict:
        """Get rate limits for a specific tier."""
        return cls.TIERS.get(tier, cls.TIERS["standard"])
    
    @classmethod
    def get_available_tiers(cls) -> list:
        """Get list of available rate limit tiers."""
        return list(cls.TIERS.keys())