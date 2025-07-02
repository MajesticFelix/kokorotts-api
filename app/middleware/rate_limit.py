"""Rate limiting middleware for KokoroTTS API.

This module provides comprehensive rate limiting functionality including:
- Request rate limiting (per minute/hour/day)
- Character count limiting for TTS requests
- Concurrent request limiting
- Redis support for distributed deployments
- In-memory storage for local development
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    # Create dummy classes for type hints when slowapi is not available
    class Limiter:
        pass
    class RateLimitExceeded(Exception):
        pass
    class SlowAPIMiddleware:
        pass

from ..config import get_config, get_client_ip, format_rate_limit_error

logger = logging.getLogger(__name__)


class CharacterCountStore:
    """Store for tracking character counts per IP."""
    
    def __init__(self, use_redis: bool = False, redis_client=None):
        self.use_redis = use_redis
        self.redis_client = redis_client
        self._memory_store: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    async def _cleanup_expired_entries(self):
        """Clean up expired entries from memory store."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        for ip in list(self._memory_store.keys()):
            ip_data = self._memory_store[ip]
            
            # Clean up expired windows
            for window in list(ip_data.keys()):
                if ":" in window:  # time-based windows like "hour:1234567890"
                    window_type, timestamp = window.split(":", 1)
                    window_age = current_time - float(timestamp)
                    
                    if window_type == "hour" and window_age > 3600:
                        del ip_data[window]
                    elif window_type == "day" and window_age > 86400:
                        del ip_data[window]
            
            # Remove empty IP entries
            if not ip_data:
                del self._memory_store[ip]
        
        self._last_cleanup = current_time
    
    async def get_character_count(self, ip: str, window: str) -> int:
        """Get character count for IP in given time window."""
        await self._cleanup_expired_entries()
        
        if self.use_redis and self.redis_client:
            try:
                key = f"char_count:{ip}:{window}"
                count = await self.redis_client.get(key)
                return int(count) if count else 0
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {e}")
                return self._memory_store.get(ip, {}).get(window, 0)
        else:
            return self._memory_store.get(ip, {}).get(window, 0)
    
    async def add_characters(self, ip: str, window: str, count: int, ttl: int = 3600):
        """Add characters to count for IP in given time window."""
        await self._cleanup_expired_entries()
        
        if self.use_redis and self.redis_client:
            try:
                key = f"char_count:{ip}:{window}"
                pipeline = self.redis_client.pipeline()
                pipeline.incrby(key, count)
                pipeline.expire(key, ttl)
                await pipeline.execute()
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {e}")
                self._memory_store[ip][window] = self._memory_store[ip].get(window, 0) + count
        else:
            self._memory_store[ip][window] = self._memory_store[ip].get(window, 0) + count


class ConcurrentRequestStore:
    """Store for tracking concurrent requests per IP."""
    
    def __init__(self):
        self._concurrent_requests: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def increment(self, ip: str) -> int:
        """Increment concurrent request count and return new count."""
        async with self._lock:
            self._concurrent_requests[ip] += 1
            return self._concurrent_requests[ip]
    
    async def decrement(self, ip: str) -> int:
        """Decrement concurrent request count and return new count."""
        async with self._lock:
            if self._concurrent_requests[ip] > 0:
                self._concurrent_requests[ip] -= 1
            count = self._concurrent_requests[ip]
            if count == 0:
                del self._concurrent_requests[ip]
            return count
    
    async def get_count(self, ip: str) -> int:
        """Get current concurrent request count for IP."""
        async with self._lock:
            return self._concurrent_requests[ip]


# Global variables for metrics tracking
_metrics_data = {
    "total_requests_blocked": 0,
    "requests_blocked_by_type": {
        "requests": 0,
        "characters": 0,
        "concurrent": 0
    },
    "startup_time": time.time()
}

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Custom rate limiting middleware for TTS-specific limits."""
    
    def __init__(
        self,
        app,
        character_store: CharacterCountStore,
        concurrent_store: ConcurrentRequestStore,
        enabled: bool = True
    ):
        super().__init__(app)
        self.character_store = character_store
        self.concurrent_store = concurrent_store
        self.enabled = enabled
        self.config = get_config()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting checks."""
        if not self.enabled:
            return await call_next(request)
        
        # Get client identifier (IP or API key based)
        client_id = get_client_identifier(request)
        # Also keep IP for backward compatibility with concurrent tracking
        client_ip = get_client_ip(request, self.config.trust_proxy_headers)
        
        # Get appropriate rate limits for this request
        rate_limits = get_rate_limits_for_request(request)
        
        # Check concurrent request limit before processing
        if request.url.path.startswith("/v1/audio/speech"):
            concurrent_count = await self.concurrent_store.increment(client_ip)
            
            try:
                # Check concurrent limit
                max_concurrent = rate_limits["max_concurrent_requests"]
                if concurrent_count > max_concurrent:
                    await self.concurrent_store.decrement(client_ip)
                    
                    # Track metrics
                    _metrics_data["total_requests_blocked"] += 1
                    _metrics_data["requests_blocked_by_type"]["concurrent"] += 1
                    
                    return JSONResponse(
                        status_code=429,
                        content=format_rate_limit_error(
                            "concurrent_requests",
                            str(max_concurrent),
                            60
                        ),
                        headers={"Retry-After": "60"}
                    )
                
                # Check character limits for TTS requests
                if request.method == "POST":
                    character_limit_response = await self._check_character_limits(request, client_id, rate_limits)
                    if character_limit_response:
                        await self.concurrent_store.decrement(client_ip)
                        return character_limit_response
                
                # Process the request
                response = await call_next(request)
                
                # Track character usage after successful request
                if request.method == "POST" and response.status_code == 200:
                    await self._track_character_usage(request, client_id)
                
                return response
            
            finally:
                # Always decrement concurrent count
                await self.concurrent_store.decrement(client_ip)
        else:
            # Non-TTS endpoints, just pass through
            return await call_next(request)
    
    async def _check_character_limits(self, request: Request, client_id: str, rate_limits: Dict[str, Any]) -> Optional[Response]:
        """Check character count limits for TTS request."""
        try:
            # Get request body to count characters
            body = await request.body()
            request._body = body  # Store for later use
            
            if not body:
                return None
            
            # Parse JSON to get input text
            try:
                request_data = json.loads(body.decode('utf-8'))
                text_input = request_data.get('input', '')
                character_count = len(text_input)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If we can't parse the body, skip character limiting
                return None
            
            # Check per-request character limit
            max_chars_per_request = rate_limits["characters_per_request"]
            if character_count > max_chars_per_request:
                # Track metrics
                _metrics_data["total_requests_blocked"] += 1
                _metrics_data["requests_blocked_by_type"]["characters"] += 1
                
                return JSONResponse(
                    status_code=429,
                    content=format_rate_limit_error(
                        "characters_per_request",
                        str(max_chars_per_request),
                        0
                    ),
                    headers={"Retry-After": "0"}
                )
            
            # Check hourly character limit
            current_time = time.time()
            hour_window = f"hour:{int(current_time // 3600)}"
            hourly_chars = await self.character_store.get_character_count(client_id, hour_window)
            
            max_chars_per_hour = rate_limits["characters_per_hour"]
            if hourly_chars + character_count > max_chars_per_hour:
                seconds_until_next_hour = 3600 - (current_time % 3600)
                
                # Track metrics
                _metrics_data["total_requests_blocked"] += 1
                _metrics_data["requests_blocked_by_type"]["characters"] += 1
                
                return JSONResponse(
                    status_code=429,
                    content=format_rate_limit_error(
                        "characters_per_hour",
                        str(max_chars_per_hour),
                        int(seconds_until_next_hour)
                    ),
                    headers={"Retry-After": str(int(seconds_until_next_hour))}
                )
            
            # Check daily character limit
            day_window = f"day:{int(current_time // 86400)}"
            daily_chars = await self.character_store.get_character_count(client_id, day_window)
            
            max_chars_per_day = rate_limits["characters_per_day"]
            if daily_chars + character_count > max_chars_per_day:
                seconds_until_next_day = 86400 - (current_time % 86400)
                
                # Track metrics
                _metrics_data["total_requests_blocked"] += 1
                _metrics_data["requests_blocked_by_type"]["characters"] += 1
                
                return JSONResponse(
                    status_code=429,
                    content=format_rate_limit_error(
                        "characters_per_day",
                        str(max_chars_per_day),
                        int(seconds_until_next_day)
                    ),
                    headers={"Retry-After": str(int(seconds_until_next_day))}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking character limits: {e}")
            return None
    
    async def _track_character_usage(self, request: Request, client_id: str):
        """Track character usage after successful request."""
        try:
            # Get the stored body
            body = getattr(request, '_body', None)
            if not body:
                return
            
            # Parse JSON to get input text
            try:
                request_data = json.loads(body.decode('utf-8'))
                text_input = request_data.get('input', '')
                character_count = len(text_input)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            
            if character_count > 0:
                current_time = time.time()
                
                # Track hourly usage
                hour_window = f"hour:{int(current_time // 3600)}"
                await self.character_store.add_characters(client_id, hour_window, character_count, 3600)
                
                # Track daily usage
                day_window = f"day:{int(current_time // 86400)}"
                await self.character_store.add_characters(client_id, day_window, character_count, 86400)
                
                logger.info(f"Tracked {character_count} characters for {client_id}")
        
        except Exception as e:
            logger.error(f"Error tracking character usage: {e}")


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting.
    
    Uses API key project name if authenticated, otherwise falls back to IP.
    This allows different rate limits for authenticated vs unauthenticated users.
    """
    # Check if request is authenticated with API key
    if hasattr(request.state, 'is_authenticated') and request.state.is_authenticated:
        project_name = getattr(request.state, 'project_name', None)
        if project_name:
            return f"api_key:{project_name}"
    
    # Fallback to IP-based rate limiting
    config = get_config()
    return f"ip:{get_client_ip(request, config.trust_proxy_headers)}"


def get_rate_limits_for_request(request: Request) -> Dict[str, Any]:
    """Get appropriate rate limits based on authentication status.
    
    Returns higher limits for API key authenticated requests,
    standard IP-based limits for unauthenticated requests.
    """
    config = get_config()
    
    # Check if request is authenticated with API key
    if hasattr(request.state, 'is_authenticated') and request.state.is_authenticated:
        rate_limit_tier = getattr(request.state, 'rate_limit_tier', 'standard')
        
        # Get API key specific limits
        api_key_limits = config.api_key.get_limits_for_tier(rate_limit_tier)
        
        return {
            "type": "api_key",
            "tier": rate_limit_tier,
            "requests_per_minute": api_key_limits.get("requests_per_minute", 100),
            "requests_per_hour": api_key_limits.get("requests_per_hour", 1000),
            "requests_per_day": api_key_limits.get("requests_per_day", 10000),
            "characters_per_request": api_key_limits.get("characters_per_request", 10000),
            "characters_per_hour": api_key_limits.get("characters_per_hour", 100000),
            "characters_per_day": api_key_limits.get("characters_per_day", 500000),
            "max_concurrent_requests": api_key_limits.get("max_concurrent_requests", 5)
        }
    else:
        # Use standard IP-based rate limits
        return {
            "type": "ip",
            "tier": "ip_based",
            "requests_per_minute": config.rate_limit.requests_per_minute,
            "requests_per_hour": config.rate_limit.requests_per_hour,
            "requests_per_day": config.rate_limit.requests_per_day,
            "characters_per_request": config.rate_limit.max_characters_per_request,
            "characters_per_hour": config.rate_limit.characters_per_hour,
            "characters_per_day": config.rate_limit.characters_per_day,
            "max_concurrent_requests": config.rate_limit.max_concurrent_requests
        }


def create_rate_limiter() -> Optional[Limiter]:
    """Create and configure rate limiter instance."""
    config = get_config()
    
    if not config.rate_limit.is_enabled() or not SLOWAPI_AVAILABLE:
        if not SLOWAPI_AVAILABLE:
            logger.warning("slowapi not available, rate limiting disabled")
        return None
    
    # Configure storage backend
    if config.rate_limit.use_redis:
        try:
            import redis.asyncio as redis
            storage_uri = config.rate_limit.redis_url
            logger.info("Using Redis for rate limiting storage")
        except ImportError:
            logger.warning("Redis not available, using in-memory storage")
            storage_uri = "memory://"
    else:
        storage_uri = "memory://"
        logger.info("Using in-memory storage for rate limiting")
    
    # Create limiter instance
    limiter = Limiter(
        key_func=get_client_identifier,
        storage_uri=storage_uri,
        default_limits=[]  # We'll apply limits via decorators
    )
    
    return limiter


def create_character_store() -> CharacterCountStore:
    """Create character count store based on configuration."""
    config = get_config()
    
    if config.rate_limit.use_redis:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(
                config.rate_limit.redis_url,
                decode_responses=True
            )
            return CharacterCountStore(use_redis=True, redis_client=redis_client)
        except ImportError:
            logger.warning("Redis not available for character store, using memory")
            return CharacterCountStore(use_redis=False)
    else:
        return CharacterCountStore(use_redis=False)


def create_concurrent_store() -> ConcurrentRequestStore:
    """Create concurrent request store."""
    return ConcurrentRequestStore()


def setup_rate_limiting(app):
    """Set up rate limiting for the FastAPI application."""
    config = get_config()
    
    if not config.rate_limit.is_enabled():
        logger.info("Rate limiting is disabled")
        return None, None, None
    
    logger.info(f"Setting up rate limiting for {config.rate_limit.deployment_mode} deployment")
    
    # Create stores
    character_store = create_character_store()
    concurrent_store = create_concurrent_store()
    
    # Create rate limiter
    limiter = create_rate_limiter()
    
    if limiter:
        # Set the limiter on app state for slowapi middleware
        app.state.limiter = limiter
        
        # Add slowapi middleware for basic rate limiting
        app.add_middleware(SlowAPIMiddleware)
        
        # Add custom middleware for character and concurrent limiting
        app.add_middleware(
            RateLimitMiddleware,
            character_store=character_store,
            concurrent_store=concurrent_store,
            enabled=True
        )
        
        # Set up rate limit exceeded handler
        @app.exception_handler(RateLimitExceeded)
        async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
            """Handle rate limit exceeded exceptions."""
            # Track metrics for slowapi rate limits
            _metrics_data["total_requests_blocked"] += 1
            _metrics_data["requests_blocked_by_type"]["requests"] += 1
            
            response = JSONResponse(
                status_code=429,
                content=format_rate_limit_error(
                    "requests",
                    str(exc.detail),
                    int(exc.retry_after) if exc.retry_after else 60
                )
            )
            if exc.retry_after:
                response.headers["Retry-After"] = str(int(exc.retry_after))
            return response
        
        logger.info("Rate limiting middleware configured successfully")
    
    return limiter, character_store, concurrent_store


async def get_rate_limit_metrics():
    """Get comprehensive rate limiting metrics."""
    config = get_config()
    
    # Basic metrics
    metrics = {
        "enabled": config.rate_limit.is_enabled(),
        "deployment_mode": config.rate_limit.deployment_mode.value,
        "total_requests_blocked": _metrics_data["total_requests_blocked"],
        "active_rate_limited_ips": 0,  # Will be calculated
        "concurrent_requests": 0,  # Will be calculated
        "character_usage_stats": {},
        "redis_connected": None
    }
    
    # Test Redis connection if enabled
    if config.rate_limit.use_redis:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(config.rate_limit.redis_url, socket_connect_timeout=2)
            await redis_client.ping()
            metrics["redis_connected"] = True
            await redis_client.close()
        except Exception:
            metrics["redis_connected"] = False
    
    # Get concurrent request counts (if stores are available)
    try:
        from . import concurrent_store
        if concurrent_store:
            # Get total concurrent requests across all IPs
            async with concurrent_store._lock:
                metrics["concurrent_requests"] = sum(concurrent_store._concurrent_requests.values())
                metrics["active_rate_limited_ips"] = len(concurrent_store._concurrent_requests)
    except Exception:
        pass
    
    # Character usage statistics
    current_time = time.time()
    hour_window = f"hour:{int(current_time // 3600)}"
    day_window = f"day:{int(current_time // 86400)}"
    
    metrics["character_usage_stats"] = {
        "current_hour_window": hour_window,
        "current_day_window": day_window,
        "total_characters_processed": 0,  # Would need persistent tracking
        "blocked_by_type": _metrics_data["requests_blocked_by_type"].copy(),
        "uptime_hours": (current_time - _metrics_data["startup_time"]) / 3600
    }
    
    # Import the metrics model from main to return proper type
    try:
        from ..main import RateLimitMetrics
        return RateLimitMetrics(**metrics)
    except ImportError:
        return metrics