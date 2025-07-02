"""API key authentication middleware for KokoroTTS API."""

import logging
import time
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import (
    get_config,
    is_api_key_authentication_enabled,
    is_api_key_required
)
from ..dependencies.auth import get_api_key_from_request, verify_api_key_in_database

logger = logging.getLogger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and request enrichment.
    
    This middleware:
    1. Extracts API keys from requests when authentication is enabled
    2. Verifies API keys against the database
    3. Adds API key info to request state for use by other middleware
    4. Tracks API key usage for authenticated requests
    """
    
    def __init__(self, app, enabled: bool = None):
        super().__init__(app)
        self.enabled = enabled if enabled is not None else is_api_key_authentication_enabled()
        self.config = get_config()
        
        # Statistics tracking
        self._stats = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "invalid_api_keys": 0,
            "database_errors": 0
        }
        
        logger.info(f"API Key Authentication Middleware initialized (enabled={self.enabled})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through API key authentication middleware."""
        start_time = time.time()
        
        # Update total request count
        self._stats["total_requests"] += 1
        
        # Initialize request state
        request.state.api_key_info = None
        request.state.is_authenticated = False
        request.state.authentication_method = "ip"  # Default to IP-based
        
        # Skip authentication if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip authentication for certain paths
        if self._should_skip_auth(request.url.path):
            return await call_next(request)
        
        try:
            # Extract API key from request
            api_key = await get_api_key_from_request(request, None)
            
            if api_key:
                # Verify API key in database
                api_key_info = await verify_api_key_in_database(api_key)
                
                if api_key_info:
                    # Successfully authenticated
                    request.state.api_key_info = api_key_info
                    request.state.is_authenticated = True
                    request.state.authentication_method = "api_key"
                    request.state.project_name = api_key_info.project_name
                    request.state.rate_limit_tier = api_key_info.rate_limit_tier
                    
                    self._stats["authenticated_requests"] += 1
                    
                    logger.debug(
                        f"API key authenticated: project={api_key_info.project_name}, "
                        f"tier={api_key_info.rate_limit_tier}"
                    )
                    
                else:
                    # Invalid API key
                    self._stats["invalid_api_keys"] += 1
                    logger.warning(f"Invalid API key attempted from {request.client.host}")
                    
                    # Store invalid key attempt info for rate limiting
                    request.state.authentication_failed = True
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Update usage statistics if authenticated
            if hasattr(request.state, 'api_key_info') and request.state.api_key_info:
                await self._update_usage_stats(request, response)
            
            # Add authentication headers to response
            self._add_auth_headers(response, request)
            
            return response
            
        except Exception as e:
            self._stats["database_errors"] += 1
            logger.error(f"Error in API key authentication middleware: {e}")
            
            # Continue without authentication on error
            return await call_next(request)
        
        finally:
            # Log request processing time
            process_time = time.time() - start_time
            logger.debug(f"API key auth middleware processed request in {process_time:.3f}s")
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if authentication should be skipped for this path."""
        skip_paths = [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/metrics",
            "/favicon.ico",
            "/static/"
        ]
        
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    async def _update_usage_stats(self, request: Request, response: Response) -> None:
        """Update usage statistics for authenticated API key."""
        try:
            api_key_info = getattr(request.state, 'api_key_info', None)
            if not api_key_info:
                return
            
            # Calculate character count from request body if TTS endpoint
            character_count = 0
            if request.url.path.startswith("/v1/audio/speech"):
                # Try to get character count from request body
                if hasattr(request.state, 'character_count'):
                    character_count = request.state.character_count
                elif hasattr(request, '_body'):
                    # Estimate from body content
                    try:
                        import json
                        body = await request.body()
                        if body:
                            data = json.loads(body)
                            if 'input' in data:
                                character_count = len(data['input'])
                    except:
                        pass
            
            # Update database usage statistics
            if character_count > 0:
                await self._update_database_usage(api_key_info.key_id, character_count)
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    async def _update_database_usage(self, key_id: str, character_count: int) -> None:
        """Update API key usage statistics in database."""
        try:
            from ..database.connection import get_session
            from ..database.models import APIKey
            
            with next(get_session()) as session:
                if session is None:
                    return
                
                # Find and update API key
                api_key = session.query(APIKey).filter(APIKey.id == key_id).first()
                if api_key:
                    api_key.update_usage(character_count)
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating database usage: {e}")
    
    def _add_auth_headers(self, response: Response, request: Request) -> None:
        """Add authentication-related headers to response."""
        if hasattr(request.state, 'is_authenticated'):
            response.headers["X-Authenticated"] = str(request.state.is_authenticated).lower()
            response.headers["X-Auth-Method"] = getattr(request.state, 'authentication_method', 'ip')
            
            if request.state.is_authenticated and hasattr(request.state, 'project_name'):
                response.headers["X-Project"] = request.state.project_name
                response.headers["X-Rate-Limit-Tier"] = getattr(request.state, 'rate_limit_tier', 'standard')
    
    def get_stats(self) -> dict:
        """Get middleware statistics."""
        return {
            **self._stats,
            "authentication_rate": (
                self._stats["authenticated_requests"] / max(self._stats["total_requests"], 1) * 100
            ),
            "error_rate": (
                self._stats["database_errors"] / max(self._stats["total_requests"], 1) * 100
            )
        }


def setup_api_key_authentication(app) -> Optional[APIKeyAuthMiddleware]:
    """Set up API key authentication middleware.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        APIKeyAuthMiddleware instance if enabled, None otherwise
    """
    config = get_config()
    
    # Only enable if API key authentication is configured
    if not is_api_key_authentication_enabled():
        logger.info("API key authentication disabled by configuration")
        return None
    
    try:
        # Initialize database connection
        from ..database.connection import ensure_database_initialized
        
        if ensure_database_initialized():
            middleware = APIKeyAuthMiddleware(app)
            app.add_middleware(APIKeyAuthMiddleware)
            logger.info("API key authentication middleware enabled")
            return middleware
        else:
            logger.warning("Database not available - API key authentication disabled")
            return None
            
    except Exception as e:
        logger.error(f"Failed to setup API key authentication: {e}")
        
        # In local development, continue without API keys
        if config.rate_limit.deployment_mode.value == "local":
            logger.info("Continuing without API key authentication in local mode")
            return None
        else:
            # In production, this might be a critical error
            raise


# Convenience function to check if request is authenticated
def is_request_authenticated(request: Request) -> bool:
    """Check if current request is authenticated with API key."""
    return getattr(request.state, 'is_authenticated', False)


def get_request_project_name(request: Request) -> Optional[str]:
    """Get project name from authenticated request."""
    if is_request_authenticated(request):
        return getattr(request.state, 'project_name', None)
    return None


def get_request_rate_limit_tier(request: Request) -> str:
    """Get rate limit tier from request (authenticated or default)."""
    if is_request_authenticated(request):
        return getattr(request.state, 'rate_limit_tier', 'standard')
    return 'ip_based'  # Default for unauthenticated requests