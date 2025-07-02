"""Pydantic models for API key management requests and responses."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from uuid import UUID


class CreateProjectRequest(BaseModel):
    """Request model for creating a new project."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Project name (alphanumeric, dashes, underscores)")
    description: Optional[str] = Field(None, max_length=500, description="Project description")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate project name format."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Project name must contain only letters, numbers, underscores, and dashes')
        return v.lower()


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key."""
    
    project_name: str = Field(..., min_length=1, max_length=100, description="Project name for the API key")
    description: Optional[str] = Field(None, max_length=500, description="API key description")
    rate_limit_tier: str = Field(default="standard", description="Rate limit tier (free, standard, premium)")
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650, description="Expiration in days (optional)")
    
    @validator('rate_limit_tier')
    def validate_tier(cls, v):
        """Validate rate limit tier."""
        allowed_tiers = ["free", "standard", "premium", "unlimited"]
        if v not in allowed_tiers:
            raise ValueError(f'Rate limit tier must be one of: {", ".join(allowed_tiers)}')
        return v


class UpdateAPIKeyRequest(BaseModel):
    """Request model for updating an API key."""
    
    description: Optional[str] = Field(None, max_length=500, description="Updated description")
    rate_limit_tier: Optional[str] = Field(None, description="Updated rate limit tier")
    is_active: Optional[bool] = Field(None, description="Whether the key is active")
    
    @validator('rate_limit_tier')
    def validate_tier(cls, v):
        """Validate rate limit tier."""
        if v is not None:
            allowed_tiers = ["free", "standard", "premium", "unlimited"]
            if v not in allowed_tiers:
                raise ValueError(f'Rate limit tier must be one of: {", ".join(allowed_tiers)}')
        return v


class APIKeyResponse(BaseModel):
    """Response model for API key information."""
    
    id: UUID = Field(..., description="API key ID")
    key_prefix: str = Field(..., description="API key prefix (for identification)")
    project_id: UUID = Field(..., description="Associated project ID")
    project_name: str = Field(..., description="Associated project name")
    description: Optional[str] = Field(None, description="API key description")
    rate_limit_tier: str = Field(..., description="Rate limit tier")
    is_active: bool = Field(..., description="Whether the key is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    usage_count: int = Field(..., description="Total usage count")
    total_characters: int = Field(..., description="Total characters processed")
    
    class Config:
        orm_mode = True


class APIKeyWithSecretResponse(APIKeyResponse):
    """Response model for API key creation (includes the actual key)."""
    
    api_key: str = Field(..., description="The actual API key (shown only once)")


class ProjectResponse(BaseModel):
    """Response model for project information."""
    
    id: UUID = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    created_at: datetime = Field(..., description="Creation timestamp")
    total_requests: int = Field(..., description="Total requests across all API keys")
    total_characters: int = Field(..., description="Total characters processed")
    api_key_count: int = Field(..., description="Number of API keys for this project")
    
    class Config:
        orm_mode = True


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys."""
    
    api_keys: List[APIKeyResponse] = Field(..., description="List of API keys")
    total_count: int = Field(..., description="Total number of API keys")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class ProjectListResponse(BaseModel):
    """Response model for listing projects."""
    
    projects: List[ProjectResponse] = Field(..., description="List of projects")
    total_count: int = Field(..., description="Total number of projects")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class APIKeyUsageStats(BaseModel):
    """Usage statistics for an API key."""
    
    api_key_id: UUID = Field(..., description="API key ID")
    project_name: str = Field(..., description="Project name")
    usage_count: int = Field(..., description="Total requests")
    total_characters: int = Field(..., description="Total characters processed")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    days_since_creation: int = Field(..., description="Days since creation")
    average_requests_per_day: float = Field(..., description="Average requests per day")
    average_characters_per_request: float = Field(..., description="Average characters per request")


class ProjectUsageStats(BaseModel):
    """Usage statistics for a project."""
    
    project_id: UUID = Field(..., description="Project ID")
    project_name: str = Field(..., description="Project name")
    total_requests: int = Field(..., description="Total requests across all API keys")
    total_characters: int = Field(..., description="Total characters processed")
    api_key_count: int = Field(..., description="Number of API keys")
    active_api_key_count: int = Field(..., description="Number of active API keys")
    days_since_creation: int = Field(..., description="Days since creation")
    average_requests_per_day: float = Field(..., description="Average requests per day")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")


class RegenerateAPIKeyResponse(BaseModel):
    """Response model for API key regeneration."""
    
    old_key_prefix: str = Field(..., description="Old API key prefix")
    new_api_key: str = Field(..., description="New API key (shown only once)")
    new_key_prefix: str = Field(..., description="New API key prefix")
    regenerated_at: datetime = Field(..., description="Regeneration timestamp")


class BulkOperationResponse(BaseModel):
    """Response model for bulk operations."""
    
    operation: str = Field(..., description="Operation performed")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class RateLimitTierInfo(BaseModel):
    """Information about a rate limit tier."""
    
    name: str = Field(..., description="Tier name")
    requests_per_minute: int = Field(..., description="Requests per minute limit")
    requests_per_hour: int = Field(..., description="Requests per hour limit")
    requests_per_day: int = Field(..., description="Requests per day limit")
    characters_per_request: int = Field(..., description="Characters per request limit")
    characters_per_hour: int = Field(..., description="Characters per hour limit")
    characters_per_day: int = Field(..., description="Characters per day limit")
    max_concurrent_requests: int = Field(..., description="Max concurrent requests")


class RateLimitTiersResponse(BaseModel):
    """Response model for available rate limit tiers."""
    
    tiers: List[RateLimitTierInfo] = Field(..., description="Available rate limit tiers")
    default_tier: str = Field(..., description="Default tier for new API keys")