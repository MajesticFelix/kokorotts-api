"""Admin router for API key management endpoints."""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, status, Query, Depends

from ..config import is_api_key_admin_enabled
from ..models.api_requests import (
    CreateProjectRequest,
    CreateAPIKeyRequest, 
    UpdateAPIKeyRequest,
    APIKeyResponse,
    APIKeyWithSecretResponse,
    ProjectResponse,
    APIKeyListResponse,
    ProjectListResponse,
    ProjectUsageStats,
    RateLimitTiersResponse,
    RateLimitTierInfo,
    RegenerateAPIKeyResponse
)
from ..utils.key_manager import KeyManager
from ..database.models import RateLimitTier

logger = logging.getLogger(__name__)

# Create admin router
admin_router = APIRouter(
    prefix="/admin/api-keys",
    tags=["Admin - API Keys"],
    responses={
        503: {"description": "Admin endpoints disabled"},
        500: {"description": "Database not available"}
    }
)


def check_admin_enabled():
    """Dependency to check if admin endpoints are enabled."""
    if not is_api_key_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "API key admin endpoints are disabled",
                    "type": "feature_disabled"
                }
            }
        )


@admin_router.get("/health", summary="Check admin API health")
async def admin_health(
    _: None = Depends(check_admin_enabled)
):
    """Check if admin API is available and database is working."""
    try:
        from ..database.connection import get_database_health
        db_health = get_database_health()
        
        return {
            "status": "healthy" if db_health["api_key_features"] else "degraded",
            "admin_enabled": True,
            "database": db_health
        }
    except Exception as e:
        logger.error(f"Admin health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Health check failed", "type": "internal_error"}}
        )


# Project Management Endpoints

@admin_router.post("/projects", response_model=ProjectResponse, summary="Create new project")
async def create_project(
    request: CreateProjectRequest,
    _: None = Depends(check_admin_enabled)
):
    """Create a new project for organizing API keys."""
    try:
        project = KeyManager.create_project(request)
        return project
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": str(e), "type": "validation_error"}}
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": str(e), "type": "database_error"}}
        )


@admin_router.get("/projects", response_model=ProjectListResponse, summary="List all projects")
async def list_projects(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    _: None = Depends(check_admin_enabled)
):
    """List all projects with pagination."""
    try:
        projects, total_count = KeyManager.list_projects(page, page_size)
        
        return ProjectListResponse(
            projects=projects,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to list projects", "type": "database_error"}}
        )


@admin_router.get("/projects/{project_name}/stats", response_model=ProjectUsageStats, summary="Get project usage statistics")
async def get_project_stats(
    project_name: str,
    _: None = Depends(check_admin_enabled)
):
    """Get detailed usage statistics for a project."""
    try:
        stats = KeyManager.get_project_usage_stats(project_name)
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"Project '{project_name}' not found", "type": "not_found"}}
            )
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to get project statistics", "type": "database_error"}}
        )


# API Key Management Endpoints

@admin_router.post("/", response_model=APIKeyWithSecretResponse, summary="Create new API key")
async def create_api_key(
    request: CreateAPIKeyRequest,
    _: None = Depends(check_admin_enabled)
):
    """Create a new API key for a project.
    
    **Warning**: The API key secret is only shown once and cannot be retrieved again.
    Make sure to save it securely.
    """
    try:
        api_key = KeyManager.create_api_key(request)
        return api_key
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": str(e), "type": "validation_error"}}
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": str(e), "type": "database_error"}}
        )


@admin_router.get("/", response_model=APIKeyListResponse, summary="List API keys")
async def list_api_keys(
    project_name: Optional[str] = Query(None, description="Filter by project name"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    _: None = Depends(check_admin_enabled)
):
    """List API keys with optional project filtering."""
    try:
        api_keys, total_count = KeyManager.list_api_keys(project_name, page, page_size)
        
        return APIKeyListResponse(
            api_keys=api_keys,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to list API keys", "type": "database_error"}}
        )


@admin_router.get("/{key_id}", response_model=APIKeyResponse, summary="Get API key details")
async def get_api_key(
    key_id: UUID,
    _: None = Depends(check_admin_enabled)
):
    """Get detailed information about a specific API key."""
    try:
        api_key = KeyManager.get_api_key(key_id)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"API key '{key_id}' not found", "type": "not_found"}}
            )
        return api_key
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to get API key", "type": "database_error"}}
        )


@admin_router.patch("/{key_id}", response_model=APIKeyResponse, summary="Update API key")
async def update_api_key(
    key_id: UUID,
    request: UpdateAPIKeyRequest,
    _: None = Depends(check_admin_enabled)
):
    """Update an API key's settings."""
    try:
        api_key = KeyManager.update_api_key(key_id, request)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"API key '{key_id}' not found", "type": "not_found"}}
            )
        return api_key
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to update API key", "type": "database_error"}}
        )


@admin_router.post("/{key_id}/regenerate", response_model=APIKeyWithSecretResponse, summary="Regenerate API key")
async def regenerate_api_key(
    key_id: UUID,
    _: None = Depends(check_admin_enabled)
):
    """Regenerate an API key (creates new secret while preserving settings).
    
    **Warning**: The old API key will be immediately invalidated and the new key 
    secret is only shown once. Make sure to update your applications with the new key.
    """
    try:
        api_key = KeyManager.regenerate_api_key(key_id)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"API key '{key_id}' not found", "type": "not_found"}}
            )
        return api_key
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to regenerate API key", "type": "database_error"}}
        )


@admin_router.delete("/{key_id}", summary="Delete API key")
async def delete_api_key(
    key_id: UUID,
    _: None = Depends(check_admin_enabled)
):
    """Delete an API key permanently.
    
    **Warning**: This action cannot be undone. The API key will be immediately 
    invalidated and all applications using it will lose access.
    """
    try:
        success = KeyManager.delete_api_key(key_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"message": f"API key '{key_id}' not found", "type": "not_found"}}
            )
        
        return {"message": "API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to delete API key", "type": "database_error"}}
        )


# Rate Limit Tier Information

@admin_router.get("/tiers", response_model=RateLimitTiersResponse, summary="Get available rate limit tiers")
async def get_rate_limit_tiers(
    _: None = Depends(check_admin_enabled)
):
    """Get information about available rate limit tiers."""
    try:
        tiers_data = KeyManager.get_available_rate_limit_tiers()
        
        tiers = []
        for tier_name, limits in tiers_data.items():
            tiers.append(RateLimitTierInfo(
                name=tier_name,
                requests_per_minute=limits["requests_per_minute"],
                requests_per_hour=limits["requests_per_hour"],
                requests_per_day=limits["requests_per_day"],
                characters_per_request=limits["characters_per_request"],
                characters_per_hour=limits["characters_per_hour"],
                characters_per_day=limits["characters_per_day"],
                max_concurrent_requests=limits["max_concurrent_requests"]
            ))
        
        return RateLimitTiersResponse(
            tiers=tiers,
            default_tier="standard"
        )
    except Exception as e:
        logger.error(f"Error getting rate limit tiers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to get rate limit tiers", "type": "internal_error"}}
        )


# Bulk Operations

@admin_router.post("/bulk/deactivate", summary="Bulk deactivate API keys")
async def bulk_deactivate_api_keys(
    project_name: str = Query(..., description="Project name to deactivate keys for"),
    _: None = Depends(check_admin_enabled)
):
    """Deactivate all API keys for a project."""
    try:
        api_keys, _ = KeyManager.list_api_keys(project_name)
        
        successful = 0
        failed = 0
        errors = []
        
        for api_key in api_keys:
            if api_key.is_active:
                update_request = UpdateAPIKeyRequest(is_active=False)
                result = KeyManager.update_api_key(api_key.id, update_request)
                if result:
                    successful += 1
                else:
                    failed += 1
                    errors.append(f"Failed to deactivate key {api_key.key_prefix}")
        
        return {
            "operation": "bulk_deactivate",
            "project_name": project_name,
            "total_keys": len(api_keys),
            "successful": successful,
            "failed": failed,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error in bulk deactivate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Bulk operation failed", "type": "database_error"}}
        )


# Statistics and Monitoring

@admin_router.get("/stats/overview", summary="Get overview statistics")
async def get_overview_stats(
    _: None = Depends(check_admin_enabled)
):
    """Get overview statistics for API key system."""
    try:
        projects, total_projects = KeyManager.list_projects(1, 1000)  # Get all projects
        api_keys, total_api_keys = KeyManager.list_api_keys(None, 1, 1000)  # Get all keys
        
        active_keys = sum(1 for key in api_keys if key.is_active)
        total_requests = sum(key.usage_count for key in api_keys)
        total_characters = sum(key.total_characters for key in api_keys)
        
        # Count keys by tier
        tier_counts = {}
        for key in api_keys:
            tier = key.rate_limit_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        return {
            "total_projects": total_projects,
            "total_api_keys": total_api_keys,
            "active_api_keys": active_keys,
            "inactive_api_keys": total_api_keys - active_keys,
            "total_requests": total_requests,
            "total_characters": total_characters,
            "keys_by_tier": tier_counts,
            "average_requests_per_key": total_requests / max(total_api_keys, 1),
            "average_characters_per_request": total_characters / max(total_requests, 1)
        }
        
    except Exception as e:
        logger.error(f"Error getting overview stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to get overview statistics", "type": "database_error"}}
        )