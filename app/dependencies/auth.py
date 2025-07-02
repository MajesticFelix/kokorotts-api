"""FastAPI dependencies for API key authentication."""

import logging
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..config import (
    get_config, 
    is_api_key_authentication_enabled,
    is_api_key_required
)
from ..utils.api_key_generator import APIKeyGenerator, validate_and_extract_project

logger = logging.getLogger(__name__)

# Security scheme for API key extraction
security = HTTPBearer(auto_error=False)


class APIKeyInfo(BaseModel):
    """Information about an authenticated API key."""
    
    key_id: str
    project_name: str
    rate_limit_tier: str
    is_authenticated: bool = True
    
    # Usage tracking
    usage_count: int = 0
    total_characters: int = 0
    
    # Metadata
    description: Optional[str] = None
    last_used_at: Optional[str] = None


async def get_api_key_from_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Extract API key from request headers.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials
        
    Returns:
        API key string if found, None otherwise
    """
    # Check Bearer token first
    if credentials and credentials.scheme == "Bearer":
        token = credentials.credentials
        if APIKeyGenerator.validate_api_key_format(token):
            return token
    
    # Check Authorization header with custom format
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("ApiKey "):
        token = auth_header.replace("ApiKey ", "").strip()
        if APIKeyGenerator.validate_api_key_format(token):
            return token
    
    # Check X-API-Key header
    api_key_header = request.headers.get("X-API-Key", "")
    if api_key_header and APIKeyGenerator.validate_api_key_format(api_key_header):
        return api_key_header
    
    return None


async def verify_api_key_in_database(api_key: str) -> Optional[APIKeyInfo]:
    """Verify API key against database and return key info.
    
    Args:
        api_key: Raw API key to verify
        
    Returns:
        APIKeyInfo if valid and active, None otherwise
    """
    try:
        from ..database.connection import get_session
        from ..database.models import APIKey, Project
        from ..utils.api_key_generator import APIKeyGenerator
        
        # Get database session
        with next(get_session()) as session:
            if session is None:
                logger.warning("Database not available for API key verification")
                return None
            
            # Get key prefix for efficient lookup
            key_prefix = APIKeyGenerator.get_key_prefix(api_key)
            if not key_prefix:
                return None
            
            # Find API key by prefix (indexed lookup)
            db_keys = session.query(APIKey).filter(
                APIKey.key_prefix == key_prefix,
                APIKey.is_active == True
            ).all()
            
            # Verify hash for matching keys
            for db_key in db_keys:
                if APIKeyGenerator.verify_api_key(api_key, db_key.key_hash):
                    # Check if key is valid (not expired)
                    if not db_key.is_valid:
                        logger.info(f"API key {key_prefix} is expired or inactive")
                        return None
                    
                    # Get project information
                    project = session.query(Project).filter(
                        Project.id == db_key.project_id
                    ).first()
                    
                    if not project:
                        logger.error(f"Project not found for API key {key_prefix}")
                        return None
                    
                    # Update usage statistics
                    db_key.update_usage(0)  # Just update last_used_at and count
                    session.commit()
                    
                    # Return key information
                    return APIKeyInfo(
                        key_id=str(db_key.id),
                        project_name=project.name,
                        rate_limit_tier=db_key.rate_limit_tier,
                        usage_count=db_key.usage_count,
                        total_characters=db_key.total_characters,
                        description=db_key.description,
                        last_used_at=db_key.last_used_at.isoformat() if db_key.last_used_at else None
                    )
            
            # No matching key found
            logger.warning(f"Invalid API key attempted: {key_prefix}")
            return None
            
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return None


async def get_api_key_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[APIKeyInfo]:
    """Optional API key authentication dependency.
    
    Returns API key info if valid key provided, None if no key or invalid key.
    Does not raise errors - suitable for endpoints that support both authenticated
    and unauthenticated access.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials
        
    Returns:
        APIKeyInfo if authenticated, None otherwise
    """
    # Check if API key authentication is enabled
    if not is_api_key_authentication_enabled():
        return None
    
    # Extract API key from request
    api_key = await get_api_key_from_request(request, credentials)
    if not api_key:
        return None
    
    # Verify API key in database
    key_info = await verify_api_key_in_database(api_key)
    if key_info:
        logger.info(f"API key authenticated: project={key_info.project_name}")
        # Store in request state for use by other middleware
        request.state.api_key_info = key_info
        return key_info
    
    logger.debug("Invalid API key provided")
    return None


async def get_api_key_required(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> APIKeyInfo:
    """Required API key authentication dependency.
    
    Raises HTTPException if no valid API key provided.
    Use this for endpoints that require authentication.
    
    Args:
        request: FastAPI request object  
        credentials: HTTP Bearer credentials
        
    Returns:
        APIKeyInfo for authenticated request
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if API key authentication is enabled
    if not is_api_key_authentication_enabled():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "error": {
                    "message": "API key authentication is not enabled",
                    "type": "feature_disabled"
                }
            }
        )
    
    # Try optional authentication first
    key_info = await get_api_key_optional(request, credentials)
    if key_info:
        return key_info
    
    # Authentication failed or no key provided
    api_key = await get_api_key_from_request(request, credentials)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "API key required. Provide via Authorization: Bearer <api_key> header",
                    "type": "authentication_required"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    else:
        # Key provided but invalid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_failed"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_authenticated_project(
    api_key_info: APIKeyInfo = Depends(get_api_key_required)
) -> str:
    """Get project name from authenticated API key.
    
    Convenience dependency that returns just the project name.
    
    Args:
        api_key_info: Authenticated API key info
        
    Returns:
        Project name string
    """
    return api_key_info.project_name


# Convenience type aliases for dependency injection
OptionalAPIKey = Annotated[Optional[APIKeyInfo], Depends(get_api_key_optional)]
RequiredAPIKey = Annotated[APIKeyInfo, Depends(get_api_key_required)]
AuthenticatedProject = Annotated[str, Depends(get_authenticated_project)]


def create_conditional_auth_dependency():
    """Create a conditional authentication dependency.
    
    Returns required authentication if API_KEY_REQUIRED=true,
    otherwise returns optional authentication.
    
    Returns:
        FastAPI dependency function
    """
    async def conditional_auth(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[APIKeyInfo]:
        if is_api_key_required():
            return await get_api_key_required(request, credentials)
        else:
            return await get_api_key_optional(request, credentials)
    
    return conditional_auth


# Default conditional dependency
ConditionalAPIKey = Annotated[
    Optional[APIKeyInfo], 
    Depends(create_conditional_auth_dependency())
]